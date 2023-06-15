recursive integer function rec(x) result(out)
    implicit none
    integer, intent(in) :: x
    
    out = rec(x+1)
end function rec

program rec1
    implicit none
    integer :: rec
    integer :: z

    z = rec(12)
end program rec1