#ifndef OPERATOR_REGISTRY_H
#define OPERATOR_REGISTRY_H

#include "ops/operator_base.h"
#include "tensor/tensor.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>


class OperatorRegistry{
    public:
        using OperatorFactory = std::function<std::unique_ptr<Operator>()>;     // 重命名一个比较复杂的类型定义,用途是把符合这个定义条件的function加载到我们后面给的名字里。


        // Now we change regostry to a global instance. 去实现 singleton
        static OperatorRegistry& instance();



        // Register an implementation.
        void registry_operator(
            const std::string& type,
            DeviceType device,
            OperatorFactory factory
        );


        // Create an operator based on its implementation
        std::unique_ptr<Operator> create(
            const std::string& type,
            DeviceType device
        ) const;


        // Judge whether the implementation has been recorded or not.
        bool contains(
            const std::string& type, 
            DeviceType device
        ) const;



        // Helper, register build-in operators at one time.
        void register_builtin_operators();



    private:

        // Set constructor as private, 这样只能被自己类内的成员进行调用。
        OperatorRegistry() = default;


        // Key = (operator type, device).
        using OperatorKey = std::pair<std::string, DeviceType>;

        
        // build a hash key generated struct for unordered map.
        struct OperatorKeyHash{

            // 定义重载（） 
            std::size_t operator()(const OperatorKey& key) const{
                const std::size_t h1 = std::hash<std::string>{}(key.first);
                const std::size_t h2 = std::hash<int>{}(static_cast<int>(key.second));

                // One calculate hash way.
                return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
            }


        };


        std::unordered_map<OperatorKey, OperatorFactory, OperatorKeyHash> factories_;



};


#endif