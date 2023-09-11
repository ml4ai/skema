program cond1
    implicit none
    integer :: x=2

    if (x < 5) then
        x = x + 1
    else
        x = x - 3
    end if
end program cond1
