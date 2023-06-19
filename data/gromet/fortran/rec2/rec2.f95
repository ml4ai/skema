recursive integer function rec1(x) result(out)
    implicit none
    integer :: rec2
    integer, intent(in) :: x
    
    out = rec2(x+1)
end function rec1

recursive integer function rec2(y) result(out)
    implicit none
    integer :: rec1
    integer, intent(in) :: y

    out = rec1(y+2)
end function rec2 

program rec2
    implicit none
    integer :: rec1
    integer :: z

    z = rec1(12)
end program rec2