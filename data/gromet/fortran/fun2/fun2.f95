integer function foo(x) 
    implicit none
    integer, intent(in):: x

    foo = x + 3
end function foo

program fun2
    implicit none
    integer :: x,y
    integer :: foo

    x = foo(2)
    y = foo(x)
end program fun2