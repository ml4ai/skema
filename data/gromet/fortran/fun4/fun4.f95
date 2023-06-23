integer function foo(x)
    implicit none
    integer, intent(in) :: x
    integer :: y, z

    y = x + 3
    z = y * 2
    foo = z
end function foo

program fun1
    implicit none
    integer :: x
    integer :: foo

    x = foo(2)
end program fun1