#include <stdlib.h>
#include <stdio.h>

// A test file to test parsing of commented quotes and
// quoted comments

int main(
    int argc,
    char* argv[])
{

    // "quotes in comment "
    char* foo = " /* comment in quotes */ ";

    /* multi-line comment
       " "
       with quotes
     */

    char* bar = " /* multi-line comment \
    in quotes */ ";

    printf("foo = %s\n", foo);
    printf("bar = %s\n", bar);

    return 0;
}

