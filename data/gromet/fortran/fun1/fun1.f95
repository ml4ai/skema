function foo(x)
    implicit none
    integer:: x
    integer :: foo

    foo = x + 2
end function foo

program fun1
    implicit none
    integer :: x
    integer :: foo

    x = foo(2)
end program fun1