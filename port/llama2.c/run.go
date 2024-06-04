package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/exp/constraints"

	"github.com/thxcode/llama-box/util/osx"
)

// ----------------------------------------------------------------------------
// Transformer model

type Config struct {
	dim        int32 // transformer dimension
	hidden_dim int32 // for ffn layers
	n_layers   int32 // number of layers
	n_heads    int32 // number of query heads
	n_kv_heads int32 // number of key/value heads (can be < n_heads because of multiquery)
	vocab_size int32 // vocabulary size, usually 256 (byte-level)
	seq_len    int32 // max sequence length
}

type TransformerWeights struct {
	// token embedding table
	token_embedding_table []float32 // (vocab_size, dim)
	// weights for rmsnorms
	rms_att_weight []float32 // (n_layers, dim) rmsnorm weights
	rms_ffn_weight []float32 // (n_layers, dim)
	// weights for matmuls. note dim == n_heads * head_size
	wq []float32 // (n_layers, dim, n_heads * head_size), head_size = dim/n_heads
	wk []float32 // (n_layers, dim, n_kv_heads * head_size)
	wv []float32 // (n_layers, dim, n_kv_heads * head_size)
	wo []float32 // (n_layers, n_heads * head_size, dim)
	// weights for ffn
	w1 []float32 // (n_layers, hidden_dim, dim)
	w2 []float32 // (n_layers, dim, hidden_dim)
	w3 []float32 // (n_layers, hidden_dim, dim)
	// final rmsnorm
	rms_final_weight []float32 // (dim,)
	// Deprecated frequency CIS for RoPE relative positional embeddings
	// freq_cis_real []float32 // (seq_len, head_size/2)
	// freq_cis_imag []float32 // (seq_len, head_size/2)
	// (optional) classifer weights for the logits, on the last layer
	wcls []float32 // (dim, vocab_size)
}

type RunState struct {
	// current wave of activations
	x      []float32 // activation at current time stamp (dim,)
	xb     []float32 // same, but inside a residual branch (dim,)
	xb2    []float32 // an additional buffer just for convenience (dim,)
	hb     []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	hb2    []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	q      []float32 // query (dim,)
	k      []float32 // key (dim,)
	v      []float32 // value (dim,)
	att    []float32 // buffer for scores/attention values (n_heads, seq_len)
	logits []float32 // output logits // (vocab_size,)
	// kv cache
	key_cache   []float32 // (layer, seq_len, kv_dim)
	value_cache []float32 // (layer, seq_len, kv_dim)
}

type Transformer struct {
	config  Config             // the hyperparameters of the architecture (the blueprint)
	weights TransformerWeights // the weights of the model
	state   RunState           // buffers for the "wave" of activations in the forward pass
	// some more state needed to properly clean up the memory mapping (sigh)
	f *osx.MmapFile
}

func malloc_run_state(s *RunState, p *Config) {
	var kv_dim int32 = p.dim * p.n_kv_heads / p.n_heads
	s.x = make([]float32, p.dim)
	s.xb = make([]float32, p.dim)
	s.xb2 = make([]float32, p.dim)
	s.hb = make([]float32, p.hidden_dim)
	s.hb2 = make([]float32, p.hidden_dim)
	s.q = make([]float32, p.dim)
	s.key_cache = make([]float32, p.n_layers*p.seq_len*kv_dim)
	s.value_cache = make([]float32, p.n_layers*p.seq_len*kv_dim)
	s.att = make([]float32, p.n_heads*p.seq_len)
	s.logits = make([]float32, p.vocab_size)
}

func free_run_state(s *RunState) {
	clear(s.x)
	clear(s.xb)
	clear(s.xb2)
	clear(s.hb)
	clear(s.hb2)
	clear(s.q)
	clear(s.att)
	clear(s.logits)
	clear(s.key_cache)
	clear(s.value_cache)
}

func memory_map_weights(w *TransformerWeights, p *Config, bs []byte, shared_weights int) {
	var head_size int32 = p.dim / p.n_heads
	// make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
	w.token_embedding_table = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.vocab_size*p.dim)
	bs = bs[p.vocab_size*p.dim*4:]
	w.rms_att_weight = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim)
	bs = bs[p.n_layers*p.dim*4:]
	w.wq = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim*(p.n_heads*head_size))
	bs = bs[p.n_layers*p.dim*(p.n_heads*head_size)*4:]
	w.wk = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim*(p.n_kv_heads*head_size))
	bs = bs[p.n_layers*p.dim*(p.n_kv_heads*head_size)*4:]
	w.wv = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim*(p.n_kv_heads*head_size))
	bs = bs[p.n_layers*p.dim*(p.n_kv_heads*head_size)*4:]
	w.wo = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*(p.n_heads*head_size)*p.dim)
	bs = bs[p.n_layers*(p.n_heads*head_size)*p.dim*4:]
	w.rms_ffn_weight = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim)
	bs = bs[p.n_layers*p.dim*4:]
	w.w1 = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.hidden_dim*p.dim)
	bs = bs[p.n_layers*p.hidden_dim*p.dim*4:]
	w.w2 = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.dim*p.hidden_dim)
	bs = bs[p.n_layers*p.dim*p.hidden_dim*4:]
	w.w3 = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.n_layers*p.hidden_dim*p.dim)
	bs = bs[p.n_layers*p.hidden_dim*p.dim*4:]
	w.rms_final_weight = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.dim)
	bs = bs[p.dim*4:]
	bs = bs[p.seq_len*head_size/2*4:] // skip what used to be freq_cis_real (for RoPE)
	bs = bs[p.seq_len*head_size/2*4:] // skip what used to be freq_cis_imag (for RoPE)
	if shared_weights > 0 {
		w.wcls = w.token_embedding_table
	} else {
		w.wcls = unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), p.dim*p.vocab_size)
		bs = bs[p.dim*p.vocab_size*4:] //nolint:staticcheck
	}
}

func read_checkpoint(checkpoint string, config *Config, weights *TransformerWeights) *osx.MmapFile {
	f, err := osx.OpenMmapFile(checkpoint)
	if err != nil {
		panic(err)
	}
	bs := f.Bytes()

	// read config
	config.dim = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.hidden_dim = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.n_layers = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.n_heads = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.n_kv_heads = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.vocab_size = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]
	config.seq_len = int32(endian.Uint32(bs[:4]))
	bs = bs[4:]

	// read weights
	var shared_weights int = ternary(config.vocab_size > 0, 1, 0)
	config.vocab_size = abs(config.vocab_size)
	memory_map_weights(weights, config, bs, shared_weights)

	return f
}

func build_transformer(t *Transformer, checkpoint string) {
	t.f = read_checkpoint(checkpoint, &t.config, &t.weights)
	malloc_run_state(&t.state, &t.config)
}

func free_transformer(t *Transformer) {
	_ = t.f.Close()
	free_run_state(&t.state)
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

func rmsnorm(o, x, weight []float32) {
	// calculate sum of squares
	var ss float32
	for i := range x {
		ss += x[i] * x[i]
	}
	ss /= float32(len(x))
	ss += 1e-5
	ss = 1 / float32(math.Sqrt(float64(ss)))
	// normalize and scale
	for i := range o {
		o[i] = weight[i] * x[i] * ss
	}
}

func softmax(x []float32) {
	// find max value (for numerical stability)
	var max_val float32 = x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max_val {
			max_val = x[i]
		}
	}
	// exp and sum
	var sum float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max_val)))
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	var (
		wg sync.WaitGroup
		n  = len(x)
	)
	for i := range xout {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var val float32
			for j := range x {
				val += w[i*n+j] * x[j]
			}
			xout[i] = val
		}(i)
	}
	wg.Wait()
}

func forward(t *Transformer, token, pos int32) []float32 {
	// a few convenience variables
	var (
		p              *Config             = &t.config
		w              *TransformerWeights = &t.weights
		s              *RunState           = &t.state
		x              []float32           = s.x
		dim            int32               = p.dim
		kv_dim         int32               = p.n_kv_heads * dim / p.n_heads
		kv_mul         int32               = p.n_heads / p.n_kv_heads // integer multiplier of the kv sharing in multiquery
		hidden_dim     int32               = p.hidden_dim
		head_size      int32               = dim / p.n_heads
		head_size_sqrt float32             = float32(math.Sqrt(float64(head_size)))
	)

	// copy the token embedding into x
	copy(x, w.token_embedding_table[token*dim:])

	// forward all the layers
	for l := int32(0); l < p.n_layers; l++ {
		/* attention */

		// attention rmsnorm
		// ?layer * dim
		rmsnorm(s.xb, x, w.rms_att_weight[l*dim:(l+1)*dim])

		// key and value point to the kv cache
		loff := l * p.seq_len * kv_dim
		s.k = s.key_cache[loff+pos*kv_dim : loff+(pos+1)*kv_dim]
		s.v = s.value_cache[loff+pos*kv_dim : loff+(pos+1)*kv_dim]

		// qkv matmuls for this position
		// ?layer * dim * n_heads * head_size =
		// ?layer * dim * n_heads * (dim / n_heads) =
		// ?layer * dim * dim
		matmul(s.q, s.xb, w.wq[l*dim*dim:(l+1)*dim*dim])
		// ?layer * dim * n_kv_heads * head_size =
		// ?layer * dim * n_kv_heads * (dim / n_heads) =
		// ?layer * dim * (n_kv_heads * dim / n_heads) =
		// ?layer * dim * kv_dim
		matmul(s.k, s.xb, w.wk[l*dim*kv_dim:(l+1)*dim*kv_dim])
		// ?layer * dim * n_kv_heads * head_size =
		// ?layer * dim * n_kv_heads * (dim / n_heads) =
		// ?layer * dim * (n_kv_heads * dim / n_heads) =
		// ?layer * dim * kv_dim
		matmul(s.v, s.xb, w.wv[l*dim*kv_dim:(l+1)*dim*kv_dim])

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for i := int32(0); i < dim; i += 2 {
			var head_dim int32 = i % head_size
			var freq float64 = 1 / math.Pow(10000, float64(head_dim)/float64(head_size))
			var val float64 = float64(pos) * freq
			var fcr, fci float64 = math.Cos(val), math.Sin(val)
			var rotn int = ternary(i < kv_dim, 2, 1) // how many vectors? 2 = q & k, 1 = q only
			for v := 0; v < rotn; v++ {
				var vec []float32 = *(ternary[*[]float32](v == 0, &s.q, &s.k))
				var v0, v1 float32 = vec[i], vec[i+1]
				vec[i] = v0*float32(fcr) - v1*float32(fci)
				vec[i+1] = v0*float32(fci) + v1*float32(fcr)
			}
		}

		// multihead attention. iterate over all heads
		{
			var wg sync.WaitGroup
			for h := int32(0); h < p.n_heads; h++ {
				wg.Add(1)
				go func(h int32) {
					defer wg.Done()
					// get the query vector for this head
					var q []float32 = s.q[h*head_size : (h+1)*head_size]
					// attention scores for this head
					var att []float32 = s.att[h*p.seq_len : (h+1)*p.seq_len]
					// iterate over all timesteps, including the current one
					for t := int32(0); t <= pos; t++ {
						// get the key vector for this head and at this timestep
						var k []float32 = s.key_cache[loff+t*kv_dim+(h/kv_mul)*head_size : loff+t*kv_dim+(h/kv_mul+1)*head_size]
						// calculate the attention score as the dot product of q and k
						var score float32
						for i := int32(0); i < head_size; i++ {
							score += q[i] * k[i]
						}
						// scale by sqrt(head_size)
						score /= head_size_sqrt
						// save the score to the attention buffer
						att[t] = score
					}

					// softmax the scores to get attention weights, from 0..pos inclusively
					softmax(att[:pos+1])

					// weighted sum of the values, store back into xb
					var xb []float32 = s.xb[h*head_size : (h+1)*head_size]
					clear(xb)
					for t := int32(0); t <= pos; t++ {
						// get the value vector for this head and at this timestep
						var v []float32 = s.value_cache[loff+t*kv_dim+(h/kv_mul)*head_size : loff+t*kv_dim+(h/kv_mul+1)*head_size]
						// accumulate the weighted value into xb
						for i := int32(0); i < head_size; i++ {
							xb[i] += att[t] * v[i]
						}
					}
				}(h)
			}
			wg.Wait()
		}

		// final matmul to get the output of the attention
		// ?layer * dim * n_heads * head_size =
		// ?layer * dim * n_heads * (dim / n_heads) =
		// ?layer * dim * dim
		matmul(s.xb2, s.xb, w.wo[l*dim*dim:(l+1)*dim*dim])

		// residual connection back into x
		for i := int32(0); i < dim; i++ {
			x[i] += s.xb2[i]
		}

		/* ffn */

		// ffn rmsnorm
		// ?layer * dim
		rmsnorm(s.xb, x, w.rms_ffn_weight[l*dim:(l+1)*dim])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))

		// first calculate self.w1(x) and self.w3(x)
		// ?layer * hidden_dim * dim
		matmul(s.hb, s.xb, w.w1[l*hidden_dim*dim:(l+1)*hidden_dim*dim])
		matmul(s.hb2, s.xb, w.w3[l*hidden_dim*dim:(l+1)*hidden_dim*dim])

		// SwiGLU non-linearity
		for i := int32(0); i < hidden_dim; i++ {
			var val float32 = s.hb[i]
			// silu(x) = x * σ(x), where σ(x) is the logistic sigmoid
			val *= 1 / (1 + float32(math.Exp(-float64(val))))
			// elementwise multiply with w3(x)
			val *= s.hb2[i]
			s.hb[i] = val
		}

		// final matmul to get the output of the FFN
		// ?layer * dim * hidden_dim
		matmul(s.xb, s.hb, w.w2[l*dim*hidden_dim:(l+1)*dim*hidden_dim])

		// residual connection back into x
		for i := int32(0); i < dim; i++ {
			x[i] += s.xb[i]
		}
	}

	// final rmsnorm
	rmsnorm(x, x, w.rms_final_weight)

	// classifier into logits
	// ?dim * vocab_size
	matmul(s.logits, x, w.wcls)
	return s.logits
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

type TokenIndex struct {
	str string
	idx int32
}

type Tokenizer struct {
	vocab            []string
	vocab_scores     []float32
	sorted_vocab     []TokenIndex
	vocab_size       int32
	max_token_length uint32
	byte_pieces      [256]byte
}

func compare_token(a, b TokenIndex) int {
	return strings.Compare(a.str, b.str)
}

func build_tokenizer(t *Tokenizer, tokenizer_path string, vocab_size int32) {
	// i should have written the vocab_size into the tokenizer file... sigh
	t.vocab_size = vocab_size
	// malloc space to hold the scores and the strings
	t.vocab = make([]string, vocab_size)
	t.vocab_scores = make([]float32, vocab_size)
	for i := 0; i < 256; i++ {
		t.byte_pieces[i] = byte(i)
	}
	// read in the file
	f, err := osx.OpenMmapFile(tokenizer_path)
	if err != nil {
		panic(err)
	}
	defer osx.Close(f)
	ff := io.NewSectionReader(f, 0, f.Len())
	fread(&t.max_token_length, ff)
	for i := int32(0); i < vocab_size; i++ {
		fread(&t.vocab_scores[i], ff)
		{
			var l int32
			fread(&l, ff)
			var b = make([]byte, l)
			fread(b, ff)
			t.vocab[i] = string(b)
		}
	}
}

func free_tokenizer(t *Tokenizer) {
	clear(t.vocab)
	clear(t.vocab_scores)
	clear(t.sorted_vocab)
}

func decode(t *Tokenizer, prev_token, token int32) string {
	// do not print <unk>, <s>, </s>
	if token <= 2 {
		return ""
	}

	var piece string = t.vocab[token]
	// following BOS (1) token, sentencepiece decoder strips any leading whitespace
	if prev_token == 1 {
		piece = strings.TrimSpace(piece)
	}
	// careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
	// parse this and convert and return the actual byte
	if piece[0] == '<' && piece[len(piece)-1] == '>' {
		var byte_val byte
		if sscanf(piece, "<0x%x>", &byte_val) == 1 {
			return string(t.byte_pieces[byte_val])
		}
	}
	return piece
}

func str_lookup(s string, sorted_vocab []TokenIndex) int32 {
	idx, found := slices.BinarySearchFunc(sorted_vocab, TokenIndex{str: s}, compare_token)
	if !found {
		return -1
	}
	return sorted_vocab[idx].idx
}

func encode(t *Tokenizer, text string, bos, eos bool) []int32 {
	var tokens = make([]int32, 0, len(text)+2) // +2 for ?BOS, ?EOS

	if t.sorted_vocab == nil {
		// lazily malloc and sort the vocabulary
		t.sorted_vocab = make([]TokenIndex, t.vocab_size)
		for i := int32(0); i < t.vocab_size; i++ {
			t.sorted_vocab[i].str = t.vocab[i]
			t.sorted_vocab[i].idx = i
		}
		slices.SortFunc(t.sorted_vocab, compare_token)
	}

	// add optional BOS (=1) token, if desired
	if bos {
		tokens = append(tokens, 1)
	}

	// add_dummy_prefix is true by default
	// so prepend a dummy prefix token to the input string, but only if text != ""
	if text != "" {
		tokens = append(tokens, str_lookup(" ", t.sorted_vocab))
	}

	// process the raw(UTF-8) byte sequence of the input string
	for _, c := range text {
		var id = str_lookup(string(c), t.sorted_vocab)
		if id != -1 {
			// we found this codepoint in vocab, add it as a token
			tokens = append(tokens, id)
			continue
		}
		// byte_fallback encoding, just encode each byte as a token
		// +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
		// so the individual bytes only start at index 3
		for _, b := range []byte(string(c)) {
			tokens = append(tokens, int32(b)+3)
		}
	}

	// merge the best consecutive pair each iteration, according the scores in vocab_socres
	for {
		var (
			best_score float32 = -1e10
			best_id    int32   = -1
			best_idx   int32   = -1
		)
		for i := int32(0); i < int32(len(tokens)-1); i++ {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			var id = str_lookup(t.vocab[tokens[i]]+t.vocab[tokens[i+1]], t.sorted_vocab)
			if id != -1 && t.vocab_scores[id] > best_score {
				// this merge pair exists in vocab! record its score and position
				best_score, best_id, best_idx = t.vocab_scores[id], id, i
			}
		}
		if best_idx == -1 {
			break // we couldn't find any more pairs to merge, so we are done
		}
		// merge the consecutive pair (best_idx, best_idx+1) into new token best_id
		tokens[best_idx] = best_id
		// shift the rest of the tokens to the left
		copy(tokens[best_idx+1:], tokens[best_idx+2:])
		tokens = tokens[:len(tokens)-1]
	}

	// add optional EOS (=2) token, if desired
	if eos {
		tokens = append(tokens, 2)
	}

	return tokens
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

type ProbIndex struct { // struct used when sorting probabilities during top-p sampling
	prob  float32
	index int32
}

type Sampler struct {
	vocab_size  int32
	probindex   []ProbIndex // buffer used in top-p sampling
	temperature float32
	topp        float32
	rng_state   int64
}

func sample_argmax(probabilities []float32) int32 {
	// return the index that has the highest probability
	var (
		max_i int32
		max_p float32 = probabilities[0]
	)
	for i, p := range probabilities {
		if p > max_p {
			max_i, max_p = int32(i), p
		}
	}
	return max_i
}

func sample_mult(probabilities []float32, coin float32) int32 {
	// sample index from probabilities (they must sum to 1!)
	// coin is a random number in [0, 1), usually from random_f32()
	var cdf float32
	for i, p := range probabilities {
		cdf += p
		if coin < cdf {
			return int32(i)
		}
	}
	return int32(len(probabilities) - 1) // in case of rounding errors
}

func compare(a, b ProbIndex) int {
	switch {
	case a.prob > b.prob:
		return -1
	case a.prob < b.prob:
		return 1
	default:
		return 0
	}
}

func sample_topp(probabilities []float32, top float32, probindex []ProbIndex, coin float32) int32 {
	// top-p sampling (or "nucleus sampling") samples from the smallest set of
	// tokens that exceed probability topp. This way we never sample tokens that
	// have very low probabilities and are less likely to go "off the rails".
	// coin is a random number in [0, 1), usually from random_f32()

	var n0 int32
	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	var cutoff float32 = (1.0 - top) / float32(len(probabilities)-1)
	for i, p := range probabilities {
		if p >= cutoff {
			probindex[n0].prob, probindex[n0].index = p, int32(i)
			n0++
		}
	}
	slices.SortFunc(probindex[:n0], compare)

	// truncate the list where cumulative probability exceeds topp
	var cumulative_prob float32
	var last_idx int32 = n0 - 1 // in case of rounding errors consider all elements
	for i := int32(0); i < n0; i++ {
		cumulative_prob += probindex[i].prob
		if cumulative_prob > top {
			last_idx = i
			break // we've execeeded topp by including last_idx
		}
	}

	// sample from the truncated list
	var r float32 = coin * cumulative_prob
	var cdf float32
	for i := int32(0); i <= last_idx; i++ {
		cdf += probindex[i].prob
		if r < cdf {
			return probindex[i].index
		}
	}
	return probindex[last_idx].index // in case of rounding errors
}

func build_sampler(s *Sampler, vocab_size int32, temperature, topp float32, rng_seed int64) {
	s.vocab_size = vocab_size
	s.probindex = make([]ProbIndex, vocab_size)
	s.temperature = temperature
	s.topp = topp
	s.rng_state = rng_seed
}

func free_sampler(s *Sampler) {
	clear(s.probindex)
}

func random_u32(state int64) uint32 {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	state ^= state >> 12
	state ^= state << 25
	state ^= state >> 27
	return uint32((state * 0x2545F4914F6CDD1D) >> 32)
}

func random_f32(state int64) float32 {
	// random float32 in [0,1)
	return float32(random_u32(state)>>8) / 16777216.0
}

func sample(sampler *Sampler, logits []float32) int32 {
	// sample the token given the logits and some hyperparameters
	var next int32
	if sampler.temperature == 0 {
		// greedy argmax sampling: take the token with the highest probability
		next = sample_argmax(logits)
	} else {
		// apply the temperature to the logits
		for q := range logits {
			logits[q] /= sampler.temperature
		}
		// apply softmax to the logits to get the probabilities for the next token
		softmax(logits)
		// flip a (float) coin (this is our source of entropy for sampling)
		var coin float32 = random_f32(sampler.rng_state)
		// we sample from this distribution to get the next token
		if sampler.topp <= 0 || sampler.topp >= 1 {
			// simply sample from the predicted probability distribution
			next = sample_mult(logits, coin)
		} else {
			// top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = sample_topp(logits, sampler.topp, sampler.probindex, coin)
		}
	}
	return next
}

// ----------------------------------------------------------------------------
// generation loop

func generate(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, prompt string, steps int32) {
	// encode the (string) prompt into tokens sequence
	var prompt_tokens = encode(tokenizer, prompt, true, false)
	var num_prompt_tokens = int32(len(prompt_tokens))

	// start the main loop
	var (
		start time.Time
		next  int32                    // will store the next token in the sequence
		token int32 = prompt_tokens[0] // kick off with the first token in the prompt
		pos   int32 = 0                // position in the sequence
	)
	for pos < steps {
		// forward the transformer to get logits for the next token
		var logits []float32 = forward(transformer, token, pos)

		// advance the state state machine
		if pos < num_prompt_tokens-1 {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos+1]
		} else {
			// otherwise, sample the next token
			next = sample(sampler, logits)
		}
		pos++

		// data-dependent terminating condition: the BOS (1) token delimits sequences
		if next == 1 {
			break
		}

		// print the token as string, decode it with the Tokenizer object
		var piece string = decode(tokenizer, token, next)
		printf("%s", piece)
		token = next

		// init the timer here because the first iteration can be slower
		if start.IsZero() {
			start = time.Now()
		}
	}
	printf("\n")

	// report achieved tok/s (pos-1 because the timer starts after first iteration)
	if pos > 1 {
		var elapsed = time.Since(start).Seconds()
		fprintf(stderr, "achieved tok/s: %f\n", float64(pos-1)/elapsed)
	}
}

func read_stdin(guide string) string {
	printf(guide)
	r, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	return strings.TrimSpace(r)
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

func chat(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, cli_user_prompt string, cli_system_prompt string, steps int32) {
	// buffers for reading the system prompt and user prompt from stdin
	var (
		system_prompt     string
		user_prompt       string
		rendered_prompt   string
		num_prompt_tokens int
		prompt_tokens     []int32
		user_idx          int
	)

	// start the main loop
	var (
		user_turn bool  = true
		next      int32     // will store the next token in the sequence
		token     int32     // stores the current token to feed into the transformer
		pos       int32 = 0 // position in the sequence
	)
	for pos < steps {
		// when it is the user's turn to contribute tokens to the dialog...
		if user_turn {
			// get the (optional) system prompt at position 0
			if pos == 0 {
				// at position 0, the user can also contribute a system prompt
				if cli_system_prompt == "" {
					// system prompt was not passed in, attempt to get if from stdin
					system_prompt = read_stdin("Enter system prompt (optional): ")
				} else {
					// system prompt was passed in, use it
					system_prompt = cli_system_prompt
				}
			}
			// get the user prompt
			if pos == 0 && cli_user_prompt != "" {
				user_prompt = cli_user_prompt
			} else {
				user_prompt = read_stdin("User: ")
			}
			// render user/system prompts into the Llama 2 Chat schema
			if pos == 0 && system_prompt != "" {
				const system_template = `[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]`
				rendered_prompt = sprintf(system_template, system_prompt, user_prompt)
			} else {
				const user_template = `[INST] %s [/INST]`
				rendered_prompt = sprintf(user_template, user_prompt)
			}
			// encode the rendered prompt into tokens
			prompt_tokens = encode(tokenizer, rendered_prompt, true, false)
			num_prompt_tokens = len(prompt_tokens)
			user_idx = 0
			user_turn = false
			printf("Assistant: ")
		}

		// determine the token to pass into the transformer next
		if user_idx < num_prompt_tokens {
			// if we are still processing the input prompt, force the next prompt token
			token = prompt_tokens[user_idx]
			user_idx++
		} else {
			// otherwise, use the next token sampled from previous turn
			token = next
		}

		// forward the transformer to get logits for the next token
		var logits []float32 = forward(transformer, token, pos)
		next = sample(sampler, logits)
		pos++

		// the Assistant is responding, so print its output
		var piece string = decode(tokenizer, token, next)
		printf("%s", piece)

		// EOS(=2) token ends the Assistant turn
		if next == 1 || next == 2 {
			user_turn = true
		}
		if user_turn {
			printf("\n")
		}

	}
	printf("\n")
}

func error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n")
	fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n")
	fprintf(stderr, "Options:\n")
	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n")
	fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n")
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n")
	fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n")
	fprintf(stderr, "  -i <string> input prompt\n")
	fprintf(stderr, "  -z <string> optional path to custom tokenizer\n")
	fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n")
	fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n")
	exit(1)
}

func main() {
	var (
		argv = os.Args
		argc = len(argv)
	)

	// default parameters
	var (
		checkpoint_path string  // e.g out/model.bin
		tokenizer_path  string  = "tokenizer.bin"
		temperature     float32 = 1.0        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
		topp            float32 = 0.9        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
		steps           int32   = 256        // number of steps to run for
		prompt          string               // prompt string
		rng_seed        int64   = 0          // seed rng with time by default
		mode            string  = "generate" // generate|chat
		system_prompt   string               // the (optional) system prompt to use in chat mode
	)

	if len(argv) >= 2 {
		checkpoint_path = argv[1]
	} else {
		error_usage()
	}
	for i := 2; i < argc; i += 2 {
		if i+1 >= argc {
			error_usage()
		}
		switch {
		case argv[i][0] != '-':
			error_usage()
		case len(argv[i]) != 2:
			error_usage()
		}
		switch argv[i][1] {
		case 't':
			temperature = atof[float32](argv[i+1])
		case 'p':
			topp = atof[float32](argv[i+1])
		case 's':
			rng_seed = atoi[int64](argv[i+1])
		case 'n':
			steps = atoi[int32](argv[i+1])
		case 'i':
			prompt = argv[i+1]
		case 'z':
			tokenizer_path = argv[i+1]
		case 'm':
			mode = argv[i+1]
		case 'y':
			system_prompt = argv[i+1]
		default:
			error_usage()
		}
	}

	// parameter validation/overrides
	if rng_seed <= 0 {
		rng_seed = int64(time.Now().Unix())
	}
	if temperature < 0.0 {
		temperature = 0.0
	}
	if topp < 0.0 || topp > 1.0 {
		topp = 0.9
	}
	if steps < 0 {
		steps = 0
	}

	// build the Transformer via the model .bin file
	var transformer Transformer
	build_transformer(&transformer, checkpoint_path)
	if steps == 0 || steps > transformer.config.seq_len {
		steps = transformer.config.seq_len // override steps to ~max length
	}
	defer free_transformer(&transformer)

	// build the Tokenizer via the tokenizer .bin file
	var tokenizer Tokenizer
	build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size)
	defer free_tokenizer(&tokenizer)

	// build the Sampler
	var sampler Sampler
	build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed)
	defer free_sampler(&sampler)

	switch mode {
	case "generate":
		generate(&transformer, &tokenizer, &sampler, prompt, steps)
	case "chat":
		chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps)
	default:
		fprintf(stderr, "unknown mode: %s\n", mode)
		error_usage()
	}
}

var (
	stderr = os.Stderr
	endian = binary.LittleEndian
)

func sscanf(s string, f string, a ...any) int {
	_, err := fmt.Sscanf(s, f, a...)
	if err != nil {
		return 0
	}
	return 1
}

func sprintf(f string, a ...any) string {
	return fmt.Sprintf(f, a...)
}

func fprintf(w io.Writer, f string, a ...any) {
	_, err := fmt.Fprintf(w, f, a...)
	if err != nil {
		panic(err)
	}
}

func printf(f string, a ...any) {
	_, err := fmt.Printf(f, a...)
	if err != nil {
		panic(err)
	}
}

func exit(c int) {
	os.Exit(c)
}

func atof[T constraints.Float](s string) T {
	r, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic(err)
	}
	return T(r)
}

func atoi[T constraints.Integer](s string) T {
	r, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		panic(err)
	}
	return T(r)
}

func fread(v any, r io.Reader) {
	err := binary.Read(r, binary.LittleEndian, v)
	if err != nil {
		panic(err)
	}
}

func ternary[T any](c bool, t, f T) T {
	if c {
		return t
	}
	return f
}

func abs[T constraints.Signed | constraints.Float](t T) T {
	if t < 0 {
		return -t
	}
	return t
}
