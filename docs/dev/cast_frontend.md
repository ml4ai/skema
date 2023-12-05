## CAST FrontEnd Generation Notes
### Using Var vs Name nodes
Currently in the CAST generation we have a convention on when to use Var and Name nodes.
The GroMEt generation depends on these being conistent, otherwise there will be errors in the generation.
In the future this convention might change, or be eliminated altogether, but for now this is the current set of rules.

- If the variable in question is being stored into (i.e. as the result of an assignment), then we use Var. 
- If the variable in question is being read from (i.e. being used in an expression), then we use Name.
