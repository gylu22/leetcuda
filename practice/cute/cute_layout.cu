#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;




int main(){

    auto parent = make_layout(make_shape(_4{},_6{}), make_stride(_6{},_1{}));
    std::cout << "Parent layout: " << parent << std::endl;
    // auto sub = make_layout(make_shape(_2{},_2{}));
    // std::cout << "Sub layout: " << sub << std::endl;
    // 除法操作
    auto result = logical_divide(parent, make_shape(_2{},_2{}));
    std::cout << "Result layout: " << result << std::endl;


    return 0;
}