program cond2
    implicit none
    integer :: x=2
    integer :: y=4

    if (x < 5) then
        x = x + y
    else
        x = x - 3
    end if
end program cond2