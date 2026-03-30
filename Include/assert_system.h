#pragma once

#include <iostream>


//ASSERT DATA STRUCT
template <typename type>
struct assert_info {
    const char* expression;
    const char* message;
    type tracked_var;
    const char* file;
    int line;
    const char* function;
};


//ASSERT SYSTEM
//because they're templates they go in the header
//only functions go in the cpp.
class assert_system {
public:
    template <typename type>
    static void check ( //nullptr is no value
        bool condition,
        const char* expression,
        const char* message,
        type tracked_var,
        const char* file, 
        int line, 
        const char* function
    ) {

        if (!condition) {
            assert_info<type> info {
                expression,
                message,
                tracked_var,
                file, 
                line, 
                function
                };

            report(info);
            abort();
        }
    }

private:
    template<typename type>
    static void report(const assert_info<type>& info){
        std::cout << "Assertion Error \n";
        std::cout << "Expression: " << info.expression << "\n";
        std::cout << info.message << "\n";
        std::cout << "Tracked Variable: " << info.tracked_var << "\n";
        std::cout << "File: \"" << info.file << "\" \n";
        std::cout << "Line: " << std::to_string(info.line) << "\n";
        std::cout << "Function: " << info.function << "\n";
    }
};


//MACROS
/**
 * @brief Terminates the program when a condition asserted to be true is not met.
 * 
 * @param cond The condition asserted to be true. eg: x != 0.
 * @param msg The message logged to the terminal if this assertion is not met.
 * 
 * @returns void. This function instead logs the expression, msg, file, line, and function, and terminates if the assertion is not met.
 */
#define ASSERT(cond, msg) \
    assert_system::check( \
        (cond),          \
        #cond,           \
        (msg),           \
        nullptr,          \
        __FILE__,        \
        __LINE__,        \
        __func__         \
    )

/**
 * @brief Terminates the program when a condition asserted to be true is not met.
 * 
 * @param cond The condition asserted to be true. eg: x != 0.
 * @param msg The message logged to the terminal if this assertion is not met.
 * @param tracking_var Prints out what this vars value is at point of failure
 * 
 * @returns void. This function instead logs the expression, msg, file, line, and function, and terminates if the assertion is not met.
 */
#define ASSERT_VAR(cond, msg, tracking_var) \
    assert_system::check( \
        (cond),          \
        #cond,           \
        (msg),           \
        (tracking_var),   \
        __FILE__,        \
        __LINE__,        \
        __func__         \
    )

#define ASSERT_ALWAYS(msg) \
    assert_system::check( \
        (true),          \
        ("always"),           \
        (msg),           \
        nullptr,          \
        __FILE__,        \
        __LINE__,        \
        __func__         \
    )


#define ASSERT_ALWAYS_VAR(msg, tracking_var) \
    assert_system::check( \
        (true),          \
        ("always"),           \
        (msg),           \
        nullptr,          \
        __FILE__,        \
        __LINE__,        \
        __func__         \
    )
//MACROS