program while3
    implicit none
    integer :: x=2
    integer :: y=3

    do while (x<5)
        x = x + 1
        x = x + y
    end do
end program while3