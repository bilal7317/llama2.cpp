#include <filesystem>

#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <iostream>
#include <string>

#include <map>
#include <iomanip>  // for std::setprecision



// Global variable for output path
std::string output_path;

// Function to read config.txt and construct output path/filename
std::string get_output_path_from_config() {
    std::string config_path = "../configs/config.txt";
    std::ifstream cfg(config_path);
    if (!cfg.is_open()) {
        throw std::runtime_error("Could not open config.txt at " + config_path);
    }
    std::string line;
    std::string results_dir, datatype, bits, zero_flag;
    bool has_results_dir = false, has_datatype = false, has_bits = false, has_zero_flag = false;
    while (std::getline(cfg, line)) {
        if (line.find("results_dir=") == 0) {
            results_dir = line.substr(line.find('=')+1);
            has_results_dir = true;
        }
        if (line.find("datatype=") == 0) {
            datatype = line.substr(line.find('=')+1);
            has_datatype = true;
        }
        if (line.find("bits=") == 0) {
            bits = line.substr(line.find('=')+1);
            has_bits = true;
        }
        if (line.find("exclude_zeros=") == 0) {
            std::string val = line.substr(line.find('=')+1);
            for (auto& c : val) c = tolower(c);
            zero_flag = (val.find("true") != std::string::npos) ? "exclude_zeros" : "include_zeros";
            has_zero_flag = true;
        }
    }
    if (!has_results_dir) throw std::runtime_error("Missing 'results_dir' in config.txt");
    if (!has_datatype) throw std::runtime_error("Missing 'datatype' in config.txt");
    if (!has_bits) throw std::runtime_error("Missing 'bits' in config.txt");
    if (!has_zero_flag) throw std::runtime_error("Missing 'exclude_zeros' in config.txt");
    std::string output_dir = results_dir + "/weights";
    std::filesystem::create_directories(output_dir);
    std::string output_filename = "output_" + datatype + "_" + bits + "_bit_" + zero_flag + ".txt";
    return output_dir + "/" + output_filename;
}


//typedef std::vector<float> tensor1d;
//typedef std::vector<__fp16> tensor1d;
//typedef std::vector<uint16_t> tensor1d;
float simulate_fp16(float x) {
    x = std::max(-5.0f, std::min(5.0f, x));
    uint32_t bits = *reinterpret_cast<uint32_t*>(&x);
    bits &= 0xFFF00000; // truncate mantissa to ~10 bits

    std::ofstream file(output_path, std::ios::app);  // Save to config-based file
    if (file.is_open()) {

        //float original = *reinterpret_cast<float*>(&bits);
        file << std::fixed << std::setprecision(8) << x << ",";  // Comma-separated
        file.close();
    } else {
        std::cerr << "Failed to open TXT file: " << output_path << "\n";
    }
    //bits &= 0xFFF00000; // truncate mantissa to ~10 bits
    
    
    // 8FE00000
    // bits &= 0xFFFFE000; // truncate mantissa to ~10 bits
    
    return *reinterpret_cast<float*>(&bits);
}


typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;



void simulate_fp16_tensor(tensor1d& t) {
    for (float& val : t) val = simulate_fp16(val);
}

void simulate_fp16_tensor(tensor2d& t) {
    for (tensor1d& row : t) simulate_fp16_tensor(row);
}

void simulate_fp16_tensor(tensor3d& t) {
    for (tensor2d& mat : t) simulate_fp16_tensor(mat);
}


float EPS = 1e-5;


struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // weights for attention matmuls
    tensor3d wq;  // [layer, dim, dim]
    tensor3d wk;  // [layer, dim, dim]
    tensor3d wv;  // [layer, dim, dim]
    tensor3d wo;  // [layer, dim, dim]
    // weights for ffn
    tensor3d w1;  // [layer, hidden_dim, dim]
    tensor3d w2;  // [layer, dim, hidden_dim]
    tensor3d w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    tensor1d rms_final_weight;  // [dim]
    // freq_cis for RoPE relatively positional embeddings
    tensor2d freq_cis_real;  // [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_imag;  // [seq_len, (dim/n_heads)/2]
};

struct RunState {
    // current wave of activations
    tensor1d x;  // activation at current time stamp [dim]
    tensor1d xb;  // same, but inside a residual branch [dim]
    tensor1d xb2;  // an additional buffer just for convenience [dim]
    tensor1d hb;  // buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d hb2;  // another buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d q;  // query [dim]
    tensor1d k;  // key [dim]
    tensor1d v;  // value [dim]
    tensor1d attention;  // buffer for scores/attention values [seq_len]
    tensor1d logits;  // buffer for logits [vocab_size]
    // kv cache
    tensor3d key_cache;  // [layer, seq_len, dim]
    tensor3d value_cache;  // [layer, seq_len, dim]
};

// --------------------------------------------------------------------------------------
// Tensor allocation and deallocation

void resize_state_tensors(RunState &state, Config &config) {
    tensor1d(config.dim).swap(state.x);
    tensor1d(config.dim).swap(state.xb);
    tensor1d(config.dim).swap(state.xb2);
    tensor1d(config.hidden_dim).swap(state.hb);
    tensor1d(config.hidden_dim).swap(state.hb2);
    tensor1d(config.dim).swap(state.q);
    tensor1d(config.dim).swap(state.k);
    tensor1d(config.dim).swap(state.v);
    tensor1d(config.seq_len).swap(state.attention);
    tensor1d(config.vocab_size).swap(state.logits);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.key_cache);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.value_cache);
}

void free_state_tensors(RunState &state) {
    state.x.clear();
    state.xb.clear();
    state.xb2.clear();
    state.hb.clear();
    state.hb2.clear();
    state.q.clear();
    state.k.clear();
    state.v.clear();
    state.attention.clear();
    state.logits.clear();
    state.key_cache.clear();
    state.value_cache.clear();
}

void resize_weights_tensors(TransformerWeights &weights, Config &config) 
{
    tensor2d(config.vocab_size, tensor1d(config.dim)).swap(weights.token_embedding_table);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_att_weight);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_ffn_weight);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wq);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wk);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wv);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wo);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w1);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim))).swap(weights.w2);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w3);
    tensor1d(config.dim).swap(weights.rms_final_weight);
    int head_size = config.dim / config.n_heads;
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_real);
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_imag);

    bool simulate_fp16_mode = true;
    if (simulate_fp16_mode) 
	{
	    simulate_fp16_tensor(weights.wq);
	    simulate_fp16_tensor(weights.wk);
	    simulate_fp16_tensor(weights.wv);
	    simulate_fp16_tensor(weights.wo);
	    simulate_fp16_tensor(weights.w1);  // ✅ correct name
	    simulate_fp16_tensor(weights.w2);  // ✅ correct name
	    simulate_fp16_tensor(weights.w3);  // ✅ correct name
	    simulate_fp16_tensor(weights.token_embedding_table);
	    simulate_fp16_tensor(weights.rms_att_weight);
	    simulate_fp16_tensor(weights.rms_ffn_weight);
	    simulate_fp16_tensor(weights.rms_final_weight);
	    // simulate_fp16_tensor(weights.wcls); ❌ remove or define if needed
	}


}

void free_weights_tensors(TransformerWeights &weights) {
    weights.token_embedding_table.clear();
    weights.rms_att_weight.clear();
    weights.rms_ffn_weight.clear();
    weights.wq.clear();
    weights.wk.clear();
    weights.wv.clear();
    weights.wo.clear();
    weights.w1.clear();
    weights.w2.clear();
    weights.w3.clear();
    weights.rms_final_weight.clear();
    weights.freq_cis_real.clear();
    weights.freq_cis_imag.clear();
}

// --------------------------------------------------------------------------------------
// Initialization: random init or read from checkpoint


// TODO: merge these into one function
void checkpoint_init_tensor(tensor1d &tensor, std::fstream &file) {
    file.read((char*)(tensor.data()), tensor.size() * sizeof(float));
}
void checkpoint_init_tensor(tensor2d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}
void checkpoint_init_tensor(tensor3d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}

void checkpoint_init_weights(TransformerWeights &weights, Config &config, std::fstream &file) {
    checkpoint_init_tensor(weights.token_embedding_table, file);
    checkpoint_init_tensor(weights.rms_att_weight, file);
    checkpoint_init_tensor(weights.wq, file);
    checkpoint_init_tensor(weights.wk, file);
    checkpoint_init_tensor(weights.wv, file);
    checkpoint_init_tensor(weights.wo, file);
    checkpoint_init_tensor(weights.rms_ffn_weight, file);
    checkpoint_init_tensor(weights.w1, file);
    checkpoint_init_tensor(weights.w2, file);
    checkpoint_init_tensor(weights.w3, file);
    checkpoint_init_tensor(weights.rms_final_weight, file);
    checkpoint_init_tensor(weights.freq_cis_real, file);
    checkpoint_init_tensor(weights.freq_cis_imag, file);
}

// --------------------------------------------------------------------------------------
// Neural net blocks

// TODO: merge these into one function
void copy(tensor1d &dst, tensor1d &src) {
    for (int i = 0; i < dst.size(); i++)  dst[i] = src[i];
}
void copy(tensor2d &dst, tensor2d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}
void copy(tensor3d &dst, tensor3d &src) {
    for (int i = 0; i < dst.size(); i++)  copy(dst[i], src[i]);
}

void accum(tensor1d &lhs, tensor1d &rhs) {
    for (int i = 0; i < rhs.size(); ++i)  lhs[i] += rhs[i];
}

void rmsnorm(tensor1d &output, tensor1d &input, tensor1d &weight) {
    float ss = 0.0;
    for (int i = 0; i < input.size(); i++)
        ss += input[i] * input[i];
    ss = ss / input.size() + EPS;
    float inv_ss = 1 / sqrt(ss);
    for (int i = 0; i < input.size(); i++)
        output[i] = input[i] * inv_ss * weight[i];
}

void softmax(tensor1d &output, tensor1d &input, int max_pos = -1) {
    if (max_pos == -1)  max_pos = input.size();
    float max_val = input[0];
    for (int i = 1; i < max_pos; i++)
        if (input[i] > max_val)  max_val = input[i];
    
    // exp and sum
    float sum = 0;
    for (int i = 0; i < max_pos; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    // normalize
    for (int i = 0; i < max_pos; i++)
        output[i] /= sum;
}

void matmul(tensor1d &output, tensor1d &input, tensor2d &weight) {
    for (int i = 0; i < output.size(); i++) {
        output[i] = 0;
        for (int j = 0; j < input.size(); j++)
            output[i] += input[j] * weight[i][j];
    }
}

void transformer(int token_index, int token_position, Config &config, RunState &state, TransformerWeights &transformer_weights) {
    // a few convenience variables
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;

    // copy the token embedding into x
    copy(state.x, transformer_weights.token_embedding_table[token_index]);
    
    simulate_fp16_tensor(state.x);
    
    for (int layer = 0; layer < config.n_layers; ++layer) {
        // attention rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_att_weight[layer]);

        // attention
        matmul(state.q, state.xb, transformer_weights.wq[layer]);
        
        simulate_fp16_tensor(state.q);
        matmul(state.k, state.xb, transformer_weights.wk[layer]);
        
        simulate_fp16_tensor(state.k);
        matmul(state.v, state.xb, transformer_weights.wv[layer]);
        
        simulate_fp16_tensor(state.v);
	
        // apply RoPE positional embeddings
        for (int head = 0; head < config.n_heads; ++head) {
            int start = head * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float q0 = state.q[start + i];
                float q1 = state.q[start + i + 1];
                float k0 = state.k[start + i];
                float k1 = state.k[start + i + 1];
                float fcr = transformer_weights.freq_cis_real[token_position][i / 2];
                float fci = transformer_weights.freq_cis_imag[token_position][i / 2];
                state.q[start + i]     = q0 * fcr - q1 * fci;
                state.q[start + i + 1] = q0 * fci + q1 * fcr;
                state.k[start + i]     = k0 * fcr - k1 * fci;
                state.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key/value in cache
        copy(state.key_cache[layer][token_position], state.k);
        copy(state.value_cache[layer][token_position], state.v);

        // multiquery attention
        for (int head = 0; head < config.n_heads; ++head) {
            for (int timestep = 0; timestep < token_position; ++timestep) {
                float score = 0;
                for (int i = 0; i < head_size; ++i)
                    score += state.q[head * head_size + i] * state.key_cache[layer][timestep][head * head_size + i];
                score /= std::sqrt(head_size * 1.0);
                state.attention[timestep] = score;
            }

            // softmax
            softmax(state.attention, state.attention, token_position+1);

            // weighted sum
            for (int i = 0; i < head_size; ++i) {
                state.xb[head * head_size + i] = 0;
                for (int timestep = 0; timestep <= token_position; ++timestep)
                    state.xb[head * head_size + i] += state.attention[timestep] * state.value_cache[layer][timestep][head * head_size + i];
            }
        }

        // final matmul to get the output of the attention
        matmul(state.xb2, state.xb, transformer_weights.wo[layer]);
	simulate_fp16_tensor(state.xb2);
        // residual connection back into x
        accum(state.x, state.xb2);

        // ffn rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_ffn_weight[layer]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x))) * self.w3(x)
        // first calculate self.w1(x) and self.w3(x)
        matmul(state.hb, state.xb, transformer_weights.w1[layer]);
        simulate_fp16_tensor(state.hb);
        matmul(state.hb2, state.xb, transformer_weights.w3[layer]);
	simulate_fp16_tensor(state.hb2);
	
        // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i])));

        // elementwise multiple with w3(x)
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * state.hb2[i];
        
        // final matmul to get the output of the ffn
        matmul(state.xb, state.hb, transformer_weights.w2[layer]);
	simulate_fp16_tensor(state.xb);
	
        // residual connection
        accum(state.x, state.xb);
    }

    // final rmsnorm
    rmsnorm(state.x, state.x, transformer_weights.rms_final_weight);

    // classifier into logits
    matmul(state.logits, state.x, transformer_weights.token_embedding_table);
}

// --------------------------------------------------------------------------------------
// Utility functions

int sample(tensor1d &probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    float r = dis(gen);

    float cdf = 0.0;
    for (int i = 0; i < probabilities.size(); ++i) {
        cdf += probabilities[i];
        if (r < cdf)
            return i;
    }
    // in case of rounding errors
    return probabilities.size() - 1;
}

int argmax(tensor1d &values) {
    int max_i = 0;
    float max_value = values[0];
    for (int i = 1; i < values.size(); ++i)
        if (values[i] > max_value) {
            max_i = i;
            max_value = values[i];
        }
    return max_i;
}

int main(int argc, char *argv[]) {
    output_path = get_output_path_from_config();
    std::cout << "Output will be saved to: " << output_path << std::endl;
    std::cout.tie(NULL);

    std::string checkpoint;
    float temperature = 0.9;
    // 'checkpoint' is a required arg
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <checkpoint_file> [temperature]\n";
        return 1;
    }
    checkpoint = argv[1];
    // temperature is optional
    if (argc >= 3)
        temperature = std::atof(argv[2]);

// Edit
std::ifstream fin("out/model.bin", std::ios::binary);
uint32_t magic;
fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
std::cout << "Magic number: " << magic << "\n";


    Config config;
    TransformerWeights transformer_weights;
    {
        std::fstream file(checkpoint);
        if (!file) {
            std::cout << "Unable to open the checkpoint file " << checkpoint << "\n";
            return 1;
        }
        // read file contents to config
        file.read((char*)&config, sizeof(config));
// Edit
        std::cout << "dim: " << config.dim << "\n";
	std::cout << "hidden_dim: " << config.hidden_dim << "\n";
	std::cout << "n_layers: " << config.n_layers << "\n";
	std::cout << "n_heads: " << config.n_heads << "\n";
	
	if (config.dim > 4096 || config.hidden_dim > 8192 || config.n_layers > 100 || 			config.n_heads > 64) {
    		std::cerr << "Config values are suspiciously large. Aborting.\n";
    		exit(1);
	}


        resize_weights_tensors(transformer_weights, config);
        checkpoint_init_weights(transformer_weights, config, file);
        file.close();
    }

    std::vector<std::string> vocab(config.vocab_size);
    {
        std::fstream file("tokenizer.bin");
        if (!file) {
            std::cout
                << "Unable to open the tokenizer file tokenizer.bin! Run \n"
                << "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n";
            return 1;
        }
        for (int i = 0; i < config.vocab_size; i++) {
            int len;
            vocab[i] = "";
            file.read((char*)&len, sizeof(int));
            for (int j = 0; j < len; ++j) {
                char c;
                file.read((char*)&c, sizeof(char));
                vocab[i].push_back(c);
            }
            vocab[i].push_back('\0');
        }
        file.close();
    }

    RunState state;
    resize_state_tensors(state, config);

    clock_t start = clock();
    int next;
    int token = 1;  // 1 = BOS token in Llama-2 sentence-piece
    for (int pos = 0; pos < config.seq_len; ++pos) {
        // forward the transformer to get logits for the next token
        transformer(token, pos, config, state, transformer_weights);

        if (temperature < EPS) {
            next = argmax(state.logits);
        } else {
            for (int q = 0; q < config.vocab_size; ++q)
                state.logits[q] /= temperature;
            softmax(state.logits, state.logits);
            next = sample(state.logits);
        }
        std::cout << vocab[next];

        token = next;
    }
    std::cout << "\n";

    // report our achieved tok/s
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("achieved tok/s: %f\n", config.seq_len / elapsed);

    // memory cleanup
    free_state_tensors(state);
    free_weights_tensors(transformer_weights);
    vocab.clear();

    return 0;
}
