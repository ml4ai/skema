#include <stdlib.h>  // an include file
#include <stdio.h>   /* another include file */

/* module level comment */

int main(
    int argc,
    char* argv[])
{

    // "quotes in comment "
    char* foo = " // comment in quotes ";

    /* quotes in multi-line
       " "
       comment
     */  int x = 5;

    char* bar = " /* multi-line comment \
    in quotes */ ";

    printf("foo = %s\n", foo);
    printf("bar = %s\n", bar);
    printf("x   = %d\n", x);

    return 0;
}
