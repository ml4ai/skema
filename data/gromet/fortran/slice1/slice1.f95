program slice1
    implicit none
    integer :: I
    integer, dimension(200) :: l=(/ (I, I=0, 199) /)
    integer, dimension(200) :: l2
    integer :: a=100
    
    
    l2 = l(4:a:5)
end program slice1