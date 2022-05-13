//
//  Timing.h
//  Deform
//
//  Created by Kyle on 11/19/18.
//  Copyright © 2018 Kyle. All rights reserved.
//
//
//  Timing.h
//  Deform
//
//  Created by Kyle on 11/19/18.
//  Copyright © 2018 Kyle. All rights reserved.
//

#ifndef Timing_h
#define Timing_h

#include <chrono>
#include <iostream>

#define TIMER_START(NAME) \
std::chrono::steady_clock::time_point __TIMER##NAME##_START = std::chrono::steady_clock::now();


#define TIMER_END(NAME) \
std::chrono::steady_clock::time_point __TIMER##NAME##_END = std::chrono::steady_clock::now(); \
{ \
auto d = std::chrono::duration_cast<std::chrono::duration<double>>( __TIMER##NAME##_END - __TIMER##NAME##_START ).count(); \
std::cout << "Timer " << #NAME << ": " << (d) << "ms" << std::endl; \
}

#endif /* Timing_h */
