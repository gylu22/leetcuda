#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;




int main(){

    auto parent = make_layout(make_shape(4,6), make_stride(6,1));
    std::cout << "Parent layout: " << parent << std::endl;
     auto sub = make_layout(make_shape(2,2), make_stride(2,1));
    std::cout << "Sub layout: " << sub << std::endl;
    // 输出: (_2,_2):(_2,_1)
    // 除法操作
   auto result = logical_divide(parent, sub);
    std::cout << "Result layout: " << result << std::endl;


    return 0;
}