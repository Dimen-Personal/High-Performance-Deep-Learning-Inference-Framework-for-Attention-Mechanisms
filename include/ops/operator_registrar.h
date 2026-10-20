// Responsible for automatic register
#ifndef OPERATOR_REGISTRAR_H
#define OPERATOR_REGISTRAR_H

#include "ops/operator_registry.h"
#include <utility>


// 相当于打包了注册的流程，不用每个函数在去写
class OperatorRegistrar{
    
    public:

        OperatorRegistrar(
            const std::string& type,
            DeviceType device,
            OperatorRegistry::OperatorFactory factory
        ){
            OperatorRegistry::instance().registry_operator(type, device, std::move(factory));    // 这里的move相当于将所有factory资源都移交给了OperatorRegistry。
        }
};



#endif