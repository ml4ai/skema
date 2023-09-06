program comments
    ! Single line comment 
    integer :: x

    ! Multi line comment
    ! Multi line comment
    integer :: y
end program comments


! Single line docstring comment
subroutine foo(a)
    integer, intent(in) :: a
    a = 1
end subroutine foo

! Multi line docstring comment 
! Multi line docstring comment
! Multi line docstring comment
! Multi line docstring comment
function bar()
    bar = 1
end function bar


! Notes on Fortran comment extraction
! 1. Comments seperated by nothing but whitespace will be considered a multi-line comments. This is due to how tree-sitter extracts comments.
! 2. The following comment types are not yet supported by tree-sitter 
!c Single line comment with 'c' character
!C Single line comment with 'C' character
!* Single line comment with '*' character
!d Single line comment with 'd' character
!D Single line comment with 'D' character