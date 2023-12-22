# -*- coding: utf-8 -*-

mml = """<math>
      <mfrac>
        <mrow>
          <mi>&#x2202;</mi>
          <mi>H</mi>
        </mrow>
        <mrow>
          <mi>&#x2202;</mi>
          <mi>t</mi>
        </mrow>
      </mfrac>
      <mo>=</mo>
      <mi>&#x2207;</mi>
      <mo>&#x22C5;</mo>
      <mo>(</mo>
      <mi>&#x0393;</mi>
      <msup>
        <mi>H</mi>
        <mrow>
          <mi>n</mi>
          <mo>+</mo>
          <mn>2</mn>
        </mrow>
      </msup>
      <mo>|</mo>
      <mi>&#x2207;</mi>
      <mi>H</mi>
      <msup>
        <mo>|</mo>
        <mrow>
          <mi>n</mi>
          <mo>&#x2212;</mo>
          <mn>1</mn>
        </mrow>
      </msup>
      <mi>&#x2207;</mi>
      <mi>H</mi>
      <mo>)</mo>
    </math>
    """

expected = 'digraph G {\n0 [color=blue, label="Div(Γ*(H^(n+2))*(Abs(Grad(H))^(n-1))*Grad(H))"];\n1 [color=blue, label="D(1, t)(H)"];\n2 [color=blue, label="Γ*(H^(n+2))*(Abs(Grad(H))^(n-1))*Grad(H)"];\n3 [color=blue, label="Γ"];\n4 [color=blue, label="H^(n+2)"];\n5 [color=blue, label="H"];\n6 [color=blue, label="n+2"];\n7 [color=blue, label="n"];\n8 [color=blue, label="2"];\n9 [color=blue, label="Abs(Grad(H))^(n-1)"];\n10 [color=blue, label="Abs(Grad(H))"];\n11 [color=blue, label="Grad(H)"];\n12 [color=blue, label="n-1"];\n13 [color=blue, label="1"];\n1 -> 0  [color=blue, label="="];\n2 -> 0  [color=blue, label="Div"];\n3 -> 2  [color=blue, label="*"];\n4 -> 2  [color=blue, label="*"];\n5 -> 4  [color=blue, label="^"];\n6 -> 4  [color=blue, label="^"];\n7 -> 6  [color=blue, label="+"];\n8 -> 6  [color=blue, label="+"];\n9 -> 2  [color=blue, label="*"];\n10 -> 9  [color=blue, label="^"];\n11 -> 10  [color=blue, label="Abs"];\n5 -> 11  [color=blue, label="Grad"];\n12 -> 9  [color=blue, label="^"];\n7 -> 12  [color=blue, label="+"];\n13 -> 12  [color=blue, label="-"];\n11 -> 2  [color=blue, label="*"];\n}\n'