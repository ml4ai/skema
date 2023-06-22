integer function foo(x) result(out)
    implicit none
    integer:: x

    out = x + 2
end function foo

program fun3
    implicit none
    integer :: x,y
    integer :: foo

    x = foo(2)
    y = foo(3)
end program fun3