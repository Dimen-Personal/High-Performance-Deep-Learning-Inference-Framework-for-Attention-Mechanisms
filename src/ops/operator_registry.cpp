#include "ops/operator_registry.h"

#include <stdexcept>
#include <utility>

void OperatorRegistry::registry_operator(
    const std::string& type,
    DeviceType device,
    OperatorFactory factory
)
{
    if(type.empty())
        throw std::invalid_argument("operator type cannot be empty.");

    if(!factory)
        throw std::invalid_argument("Operator Factory cannot be empty.");

    
    OperatorKey key = {type, device};


    // Check the factory has already existed or not.
    // emplace will insert a new pair into the container and it will not replace old one if confliction existed.
    // Result 第一层指向迭代器，若插入成功则迭代器指向新的pair,若失败则迭代器指向已经存在的pair，在迭代器中可以使用first 和 second来进行数据的提取。
    // 第二层指向的就是插入结果为bool类型。
    auto result = factories_.emplace(std::move(key), std::move(factory));

    if(!result.second)
        throw std::runtime_error("Operator already registered: " + type);
}




std::unique_ptr<Operator> OperatorRegistry::create(
    const std::string& type,
    DeviceType device
) const
{
    OperatorKey key = {type, device};

    auto it = factories_.find(key);

    if(it == factories_.end())
        throw std::runtime_error("Operator implementation not existed");

    return it -> second(); // 这里最后一步（）是因为我们second找到的是一个函数，我们要让这个函数去运行起来。
}




bool OperatorRegistry::contains(const std::string& type, DeviceType device) const{
    OperatorKey key = {type, device};

    return factories_.find(key) != factories_.end();
}



// Meters Singleton
OperatorRegistry& OperatorRegistry::instance(){
    static OperatorRegistry registry;    // static 局部变量只会在第一次运行的时候创建一次然后后面再次调用就会返回已经创建的实例。
    return registry;
}