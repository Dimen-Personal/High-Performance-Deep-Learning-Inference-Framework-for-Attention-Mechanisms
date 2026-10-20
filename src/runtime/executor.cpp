#include "runtime/executor.h"
#include "ops/operator.h"
#include "cuda/cuda_ops.cuh"
#include <stdexcept>

void Executor::add_tensor(const Tensor& tensor){
    tensors_.insert_or_assign(tensor.name(), tensor);  // C++ dunction for hash map instance.
    tensor_shapes_.insert_or_assign(tensor.name(), tensor.shape());
    tensor_devices_.insert_or_assign(tensor.name(), tensor.device());
}

void Executor::register_tensor(const std::string& name, const std::vector<int>& shape, DeviceType device){
    tensor_shapes_.insert_or_assign(name, shape);
    tensor_devices_.insert_or_assign(name, device);
}


void Executor::allocate_tensor_if_needed(const std::string& name){
    // Already exists.
    if(tensors_.find(name) != tensors_.end()) return;



    auto shape_it = tensor_shapes_.find(name);
    // This tensor not been registered.
    if(shape_it == tensor_shapes_.end())
        throw std::runtime_error("Tensor shape not registered: " + name);


    
    // Add location judge.
    auto device_it = tensor_devices_.find(name);
    if(device_it == tensor_devices_.end())
        throw std::runtime_error("Tensor device not registered: " + name);


    const std::vector<int>& shape = shape_it -> second;
    const DeviceType device = device_it -> second;


    std::size_t size = 1;
    for(int dim : shape)
        size *= static_cast<std::size_t>(dim);



    // Starts allocation.
    float* buffer =  nullptr;

    if(device == DeviceType::CPU)
        buffer = memory_pool_.allocate(size);
    else
        buffer = cuda_memory_pool_.allocate(size);

    Tensor tensor(name, shape, buffer, device);
    tensors_.insert_or_assign(name, tensor);
     
    
}



// Tensor& 相当于创建一个名字，进行引用，而不是直接复制一份
Tensor& Executor::get_tensor(const std::string& name){
    auto it = tensors_.find(name);  // If directly use tensor_[name] at here, if name doesn't exists, map will create an empty instance immediately.

    if(it == tensors_.end())
        throw std::runtime_error("Tensor not found: " + name);

    return it -> second;
}
 


const Tensor& Executor::get_tensor(const std::string& name) const{
    auto it = tensors_.find(name);

    if(it == tensors_.end())
        throw std::runtime_error("Tensor not found: " + name);

    return it -> second;
}






// Running Helper
void Executor::release_tensor_if_dead(const std::string& name, int step){
    if(!memory_planner_.is_intermediate(name))
        return;

    const TensorLifetime& lifetime = memory_planner_.lifetime(name);

    if(lifetime.last_use != step)
        return;

    
    auto it = tensors_.find(name);
    if(it == tensors_.end())
        return;

    
    Tensor& tensor = it -> second;
    if(tensor.owns_memory())
        return;


    if(tensor.device() == DeviceType::CPU)
        memory_pool_.release(tensor.data());
    else
        cuda_memory_pool_.release(tensor.data());




    // Delete this map in the hash maps.
    tensors_.erase(it);

}





void Executor::run(const Graph& graph){
    memory_planner_.analyze(graph);
    std::vector<int> order = graph.topological_sort();

    for(int step = 0; step < static_cast<int>(order.size()); step ++){
        int node_index = order[step];
        const Node& node = graph.nodes()[node_index];

        // Allocate output tensors. (all inputs 肯定被处理好了)
        for(const std::string& output : node.outputs())
            allocate_tensor_if_needed(output);


        // Execute operator
        execute_node(node);


        
        // Release dead input tensors
        for(const std::string& input : node.inputs())
            release_tensor_if_dead(input, step);



    }
}



/*
OperatorRegistry& Executor::operator_registry(){
    return operator_registry_;
}

const OperatorRegistry& Executor::operator_registry() const{
    return operator_registry_;
}


*/







// Next will be large execute node part.
void Executor::execute_node(const Node& node){
    
    // Load inputs tesnor
    std::vector<const Tensor*> inputs;
    inputs.reserve(node.inputs().size());    // reserve 是预留空间，避免后面超出容量后进行整块的迁移和重新定位    // 注意这边size返回的是元素数量

    for(const std::string& input_name : node.inputs())
        inputs.push_back(&get_tensor(input_name));



    // Load output tensors
    std::vector<Tensor*> outputs;
    outputs.reserve(node.outputs().size());
    for(const std::string& output_name : node.outputs())
        outputs.push_back(&get_tensor(output_name));


    
    // Determine Device
    if(inputs.empty() && outputs.empty())
        throw std::runtime_error("Operator has no tensors.");

    DeviceType device = !inputs.empty() ? inputs[0] -> device() : outputs[0] -> device();





    // Validate device consistence.
    for(const Tensor* input : inputs)
        if(input -> device() != device)
            throw std::runtime_error("Input Device Mismatch.");

    for(const Tensor* output : outputs)
        if(output -> device() != device)
            throw std::runtime_error("Output Device mismatch.");


    

    // Create operator
    //std::unique_ptr<Operator> op = operator_registry_.create(node.type(), device);
    auto op = OperatorRegistry::instance().create(node.type(), device);


    // Execute
    op -> execute(inputs, outputs);
    
    
    
    
    // if(node.type() == "AttentionSoftmax"){
    //     execute_node_attention_softmax(node);
    //     return;
    // }
        
    
    // if(node.type() == "MatMul"){
    //     execute_matmul(node);
    //     return;
    // }
    
    
    // if(node.type() == "GELU"){
    //     execute_gelu(node);
    //     return;
    // }
    
    
    
    // if(node.type() == "LayerNorm"){
    //     execute_layernorm(node);
    //     return;
    // }


    // if(node.type() == "AttentionScores"){
    //     execute_attention_scores(node);
    //     return;
    // }
    
    
    //  throw std::runtime_error("Unsupported operator type: " + node.type());


 }





// Two exposure tests interface
std::size_t Executor::memory_block_count() const{
    return memory_pool_.block_count();
}


std::size_t Executor::memory_capacity_count() const{
    return memory_pool_.total_capacity();
}


// void Executor::execute_attention_scores(const Node& node){
//     if(node.inputs().size() != 2 || node.outputs().size() != 1)
//         throw std::runtime_error("AttentionScores expects 2 inputs and 1 output.");
    

//     Tensor& Q = get_tensor(node.inputs()[0]);
//     Tensor& K = get_tensor(node.inputs()[1]);
//     Tensor& scores = get_tensor(node.outputs()[0]);


//     // Q and K must be 2D
//     if(Q.shape().size() != 2 || K.shape().size() != 2 || scores.shape().size() != 2)
//         throw std::runtime_error("AttentionScores requires 2D tensors.");
    


//     int seq_len = Q.shape()[0];
//     int head_dim = Q.shape()[1];


//     // Q and K should have same shape
//     if(K.shape()[0] != seq_len || K.shape()[1] != head_dim)
//         throw std::runtime_error("Q and K shape mismatch.");
    


//     // scores should be seq_len × seq_len
//     if(scores.shape()[0] != seq_len || scores.shape()[1] != seq_len)
//         throw std::runtime_error("Attention scores output shape mismatch.");
    

//     /*
//     attention_scores(
//         Q.data(),
//         K.data(),
//         scores.data(),
//         seq_len,
//         head_dim
//     );
//     */

//     if(Q.device() != K.device() || Q.device() != scores.device())
//         throw std::runtime_error("AttentionScores Tensors must be on the same device.");

    
//     if(Q.device() == DeviceType::CPU)
//         attention_scores(Q.data(), K.data(), scores.data(), seq_len, head_dim);
//     else
//         cuda_attention_scores(Q.data(), K.data(), scores.data(), seq_len, head_dim);

// }



// void Executor::execute_layernorm(const Node& node){
//     if(node.inputs().size() != 3 || node.outputs().size() != 1)
//         throw std::runtime_error("LayerNorm expects input, gamma, beta and one output");

//     Tensor& input = get_tensor(node.inputs()[0]);
//     Tensor& gamma = get_tensor(node.inputs()[1]);
//     Tensor& beta = get_tensor(node.inputs()[2]);
//     Tensor& output = get_tensor(node.outputs()[0]);

//     if(input.size() != gamma.size() || input.size() != beta.size() || input.size() != output.size())
//         throw std::runtime_error("LayerNorm tensor size mismatch.");

//     if(input.device() != gamma.device() || input.device() != beta.device() || input.device() != output.device())
//         throw std::runtime_error("LayerNorm tensors must be on the same device.");



//     constexpr float epsilon = 1e-5;  // constexpr means fix value on the compile state and never change later.

//     //layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), epsilon);

//     if(input.device() == DeviceType::CPU)
//         layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), epsilon);
//     else
//         cuda_layer_norm(input.data(), gamma.data(), beta.data(), output.data(), static_cast<int>(input.size()), epsilon);

// }



// void Executor::execute_gelu(const Node& node){
//     if(node.inputs().size() != 1 || node.outputs().size() != 1)
//         throw std::runtime_error("GELU exoects 1 input and 1 output.");


//     Tensor& input = get_tensor(node.inputs()[0]);
//     Tensor& output = get_tensor(node.outputs()[0]);

    
//     if(input.shape() != output.shape())
//         throw std::runtime_error("GRLU input and output mismatch.");

//     if(input.device() != output.device())
//         throw std::runtime_error("GELU input and output must be on the same device");


//     if(input.device() == DeviceType::CPU)
//         gelu(input.data(), output.data(), static_cast<int>(input.size()));
//     else
//         cuda_gelu(input.data(), output.data(), static_cast<int>(input.size()));
     
// }




// void Executor::execute_matmul(const Node& node){
//     if(node.inputs().size() != 2 || node.outputs().size() != 1)
//         throw std::runtime_error("Matmul expects 2 inputs and 1 output.");

//     Tensor& A = get_tensor(node.inputs()[0]);
//     Tensor& B = get_tensor(node.inputs()[1]);
//     Tensor& C = get_tensor(node.outputs()[0]);


//     // Cuurrently, out matmul only works for 2D.
//     if(A.shape().size() !=2 || B.shape().size() != 2 || C.shape().size() != 2)
//         throw std::runtime_error("Matmul requires 2D Tensors.");

//     int M = A.shape()[0];
//     int K = A.shape()[1];
//     int B_K = B.shape()[0];
//     int N = B.shape()[1];

//     // A: M x K   B: K x N
//     if(K != B_K)
//         throw std::runtime_error("matmul input shape mismatch");
    
//     // C: M x N
//     if(C.shape()[0] != M || C.shape()[1] != N)
//         throw std::runtime_error("matmul output shape mismatch");

//     //matmul(A.data(), B.data(), C.data(), M, K, N);

//     if(A.device() != B.device() || A.device() != C.device())
//         throw std::runtime_error("Matmul tensors must be on the same device.");

//     if(A.device() == DeviceType::CPU)
//         matmul(A.data(), B.data(), C.data(), M, K, N);
//     else
//         cuda_matmul(A.data(), B.data(), C.data(), M, K , N);
    
// }




// void Executor::execute_node_attention_softmax(const Node& node){
//     if(node.inputs().size() != 1 || node.outputs().size() != 1)
//         throw std::runtime_error("Softmax node Exepects 1 input and 1 output");

//     Tensor& input = get_tensor(node.inputs()[0]);
//     Tensor& output = get_tensor(node.outputs()[0]);

//     if(input.size() != output.size())
//         throw std::runtime_error("Softmax input/output size mismatch.");

//     if(input.shape() != output.shape())
//         throw std::runtime_error("Softmax input/output shape mismatch.");

//     if(input.shape().size() != 2)
//         throw std::runtime_error("Attention softmax must be square.");


//     if(input.shape()[0] != input.shape()[1])
//         throw std::runtime_error("Attention softmax input must be square.");

   

//     int seq_len = input.shape()[0];
    
//     //attention_softmax(input.data(), output.data(), static_cast<int>(seq_len));

//     if(input.device() != output.device())
//         throw std::runtime_error("Attention Softmax Tensors must be on the same device.");

//     if(input.device() == DeviceType::CPU)
//         attention_softmax(input.data(), output.data(), seq_len);
//     else
//         cuda_attention_softmax(input.data(), output.data(), seq_len);
// }


 

