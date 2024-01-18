(function() {var type_impls = {
"zstd_safe":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-CCtx%3C'a%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#224-809\">source</a><a href=\"#impl-CCtx%3C'a%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a&gt; <a class=\"struct\" href=\"zstd_safe/struct.CCtx.html\" title=\"struct zstd_safe::CCtx\">CCtx</a>&lt;'a&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.try_create\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#228-234\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.try_create\" class=\"fn\">try_create</a>() -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;Self&gt;</h4></section></summary><div class=\"docblock\"><p>Tries to create a new context.</p>\n<p>Returns <code>None</code> if zstd returns a NULL pointer - may happen if allocation fails.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.create\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#241-244\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.create\" class=\"fn\">create</a>() -&gt; Self</h4></section></summary><div class=\"docblock\"><p>Wrap <code>ZSTD_createCCtx</code></p>\n<h5 id=\"panics\"><a href=\"#panics\">Panics</a></h5>\n<p>If zstd returns a NULL pointer.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#247-266\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress\" class=\"fn\">compress</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    dst: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.reference.html\">&amp;mut C</a>,\n    src: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>],\n    compression_level: <a class=\"type\" href=\"zstd_safe/type.CompressionLevel.html\" title=\"type zstd_safe::CompressionLevel\">CompressionLevel</a>\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Wraps the <code>ZSTD_compressCCtx()</code> function</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress2\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#269-286\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress2\" class=\"fn\">compress2</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    dst: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.reference.html\">&amp;mut C</a>,\n    src: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>]\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Wraps the <code>ZSTD_compress2()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress_using_dict\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#289-311\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress_using_dict\" class=\"fn\">compress_using_dict</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    dst: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.reference.html\">&amp;mut C</a>,\n    src: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>],\n    dict: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>],\n    compression_level: <a class=\"type\" href=\"zstd_safe/type.CompressionLevel.html\" title=\"type zstd_safe::CompressionLevel\">CompressionLevel</a>\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Wraps the <code>ZSTD_compress_usingDict()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress_using_cdict\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#314-333\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress_using_cdict\" class=\"fn\">compress_using_cdict</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    dst: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.reference.html\">&amp;mut C</a>,\n    src: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>],\n    cdict: &amp;<a class=\"struct\" href=\"zstd_safe/struct.CDict.html\" title=\"struct zstd_safe::CDict\">CDict</a>&lt;'_&gt;\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Wraps the <code>ZSTD_compress_usingCDict()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.init\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#340-346\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.init\" class=\"fn\">init</a>(&amp;mut self, compression_level: <a class=\"type\" href=\"zstd_safe/type.CompressionLevel.html\" title=\"type zstd_safe::CompressionLevel\">CompressionLevel</a>) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Initializes the context with the given compression level.</p>\n<p>This is equivalent to running:</p>\n<ul>\n<li><code>reset()</code></li>\n<li><code>set_parameter(CompressionLevel, compression_level)</code></li>\n</ul>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.load_dictionary\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#416-425\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.load_dictionary\" class=\"fn\">load_dictionary</a>(&amp;mut self, dict: &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>]) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Tries to load a dictionary.</p>\n<p>The dictionary content will be copied internally and does not need to be kept alive after\ncalling this function.</p>\n<p>If you need to use the same dictionary for multiple contexts, it may be more efficient to\ncreate a <code>CDict</code> first, then loads that.</p>\n<p>The dictionary will apply to all compressed frames, until a new dictionary is set.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ref_cdict\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#430-438\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.ref_cdict\" class=\"fn\">ref_cdict</a>&lt;'b&gt;(&amp;mut self, cdict: &amp;<a class=\"struct\" href=\"zstd_safe/struct.CDict.html\" title=\"struct zstd_safe::CDict\">CDict</a>&lt;'b&gt;) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a><span class=\"where fmt-newline\">where\n    'b: 'a,</span></h4></section></summary><div class=\"docblock\"><p>Wraps the <code>ZSTD_CCtx_refCDict()</code> function.</p>\n<p>Dictionary must outlive the context.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.disable_dictionary\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#443-452\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.disable_dictionary\" class=\"fn\">disable_dictionary</a>(&amp;mut self) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Return to “no-dictionary” mode.</p>\n<p>This will disable any dictionary/prefix previously registered for future frames.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ref_prefix\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#459-471\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.ref_prefix\" class=\"fn\">ref_prefix</a>&lt;'b&gt;(&amp;mut self, prefix: &amp;'b [<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u8.html\">u8</a>]) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a><span class=\"where fmt-newline\">where\n    'b: 'a,</span></h4></section></summary><div class=\"docblock\"><p>Use some prefix as single-use dictionary for the next compressed frame.</p>\n<p>Just like a dictionary, decompression will need to be given the same prefix.</p>\n<p>This is best used if the “prefix” looks like the data to be compressed.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress_stream\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#484-500\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress_stream\" class=\"fn\">compress_stream</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    output: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.OutBuffer.html\" title=\"struct zstd_safe::OutBuffer\">OutBuffer</a>&lt;'_, C&gt;,\n    input: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.InBuffer.html\" title=\"struct zstd_safe::InBuffer\">InBuffer</a>&lt;'_&gt;\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Performs a step of a streaming compression operation.</p>\n<p>This will read some data from <code>input</code> and/or write some data to <code>output</code>.</p>\n<h5 id=\"returns\"><a href=\"#returns\">Returns</a></h5>\n<p>A hint for the “ideal” amount of input data to provide in the next call.</p>\n<p>This hint is only for performance purposes.</p>\n<p>Wraps the <code>ZSTD_compressStream()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.compress_stream2\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#517-534\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.compress_stream2\" class=\"fn\">compress_stream2</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    output: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.OutBuffer.html\" title=\"struct zstd_safe::OutBuffer\">OutBuffer</a>&lt;'_, C&gt;,\n    input: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.InBuffer.html\" title=\"struct zstd_safe::InBuffer\">InBuffer</a>&lt;'_&gt;,\n    end_op: <a class=\"enum\" href=\"zstd_sys/enum.ZSTD_EndDirective.html\" title=\"enum zstd_sys::ZSTD_EndDirective\">ZSTD_EndDirective</a>\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Performs a step of a streaming compression operation.</p>\n<p>This will read some data from <code>input</code> and/or write some data to <code>output</code>.</p>\n<p>The <code>end_op</code> directive can be used to specify what to do after: nothing special, flush\ninternal buffers, or end the frame.</p>\n<h5 id=\"returns-1\"><a href=\"#returns-1\">Returns</a></h5>\n<p>An lower bound for the amount of data that still needs to be flushed out.</p>\n<p>This is useful when flushing or ending the frame: you need to keep calling this function\nuntil it returns 0.</p>\n<p>Wraps the <code>ZSTD_compressStream2()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.flush_stream\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#541-551\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.flush_stream\" class=\"fn\">flush_stream</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    output: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.OutBuffer.html\" title=\"struct zstd_safe::OutBuffer\">OutBuffer</a>&lt;'_, C&gt;\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Flush any intermediate buffer.</p>\n<p>To fully flush, you should keep calling this function until it returns <code>Ok(0)</code>.</p>\n<p>Wraps the <code>ZSTD_flushStream()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.end_stream\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#558-568\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.end_stream\" class=\"fn\">end_stream</a>&lt;C: <a class=\"trait\" href=\"zstd_safe/trait.WriteBuf.html\" title=\"trait zstd_safe::WriteBuf\">WriteBuf</a> + ?<a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>&gt;(\n    &amp;mut self,\n    output: &amp;mut <a class=\"struct\" href=\"zstd_safe/struct.OutBuffer.html\" title=\"struct zstd_safe::OutBuffer\">OutBuffer</a>&lt;'_, C&gt;\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Ends the stream.</p>\n<p>You should keep calling this function until it returns <code>Ok(0)</code>.</p>\n<p>Wraps the <code>ZSTD_endStream()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.sizeof\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#573-576\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.sizeof\" class=\"fn\">sizeof</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the size currently used by this context.</p>\n<p>This may change over time.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.reset\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#583-588\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.reset\" class=\"fn\">reset</a>(&amp;mut self, reset: <a class=\"enum\" href=\"zstd_safe/enum.ResetDirective.html\" title=\"enum zstd_safe::ResetDirective\">ResetDirective</a>) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Resets the state of the context.</p>\n<p>Depending on the reset mode, it can reset the session, the parameters, or both.</p>\n<p>Wraps the <code>ZSTD_CCtx_reset()</code> function.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.set_parameter\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#593-715\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.set_parameter\" class=\"fn\">set_parameter</a>(&amp;mut self, param: <a class=\"enum\" href=\"zstd_safe/enum.CParameter.html\" title=\"enum zstd_safe::CParameter\">CParameter</a>) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Sets a compression parameter.</p>\n<p>Some of these parameters need to be set during de-compression as well.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.set_pledged_src_size\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#725-736\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.set_pledged_src_size\" class=\"fn\">set_pledged_src_size</a>(\n    &amp;mut self,\n    pledged_src_size: <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.u64.html\">u64</a>&gt;\n) -&gt; <a class=\"type\" href=\"zstd_safe/type.SafeResult.html\" title=\"type zstd_safe::SafeResult\">SafeResult</a></h4></section></summary><div class=\"docblock\"><p>Guarantee that the input size will be this value.</p>\n<p>If given <code>None</code>, assumes the size is unknown.</p>\n<p>Unless explicitly disabled, this will cause the size to be written in the compressed frame\nheader.</p>\n<p>If the actual data given to compress has a different size, an error will be returned.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.in_size\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#797-800\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.in_size\" class=\"fn\">in_size</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the recommended input buffer size.</p>\n<p>Using this size may result in minor performance boost.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.out_size\" class=\"method\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#805-808\">source</a><h4 class=\"code-header\">pub fn <a href=\"zstd_safe/struct.CCtx.html#tymethod.out_size\" class=\"fn\">out_size</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the recommended output buffer size.</p>\n<p>Using this may result in minor performance boost.</p>\n</div></details></div></details>",0,"zstd_safe::CStream"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Default-for-CCtx%3C'_%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#218-222\">source</a><a href=\"#impl-Default-for-CCtx%3C'_%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> for <a class=\"struct\" href=\"zstd_safe/struct.CCtx.html\" title=\"struct zstd_safe::CCtx\">CCtx</a>&lt;'_&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.default\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#219-221\">source</a><a href=\"#method.default\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.75.0/core/default/trait.Default.html#tymethod.default\" class=\"fn\">default</a>() -&gt; Self</h4></section></summary><div class='docblock'>Returns the “default value” for a type. <a href=\"https://doc.rust-lang.org/1.75.0/core/default/trait.Default.html#tymethod.default\">Read more</a></div></details></div></details>","Default","zstd_safe::CStream"],["<section id=\"impl-Send-for-CCtx%3C'a%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#820\">source</a><a href=\"#impl-Send-for-CCtx%3C'a%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"zstd_safe/struct.CCtx.html\" title=\"struct zstd_safe::CCtx\">CCtx</a>&lt;'a&gt;</h3></section>","Send","zstd_safe::CStream"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Drop-for-CCtx%3C'a%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#811-818\">source</a><a href=\"#impl-Drop-for-CCtx%3C'a%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/ops/drop/trait.Drop.html\" title=\"trait core::ops::drop::Drop\">Drop</a> for <a class=\"struct\" href=\"zstd_safe/struct.CCtx.html\" title=\"struct zstd_safe::CCtx\">CCtx</a>&lt;'a&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.drop\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/zstd_safe/lib.rs.html#812-817\">source</a><a href=\"#method.drop\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.75.0/core/ops/drop/trait.Drop.html#tymethod.drop\" class=\"fn\">drop</a>(&amp;mut self)</h4></section></summary><div class='docblock'>Executes the destructor for this type. <a href=\"https://doc.rust-lang.org/1.75.0/core/ops/drop/trait.Drop.html#tymethod.drop\">Read more</a></div></details></div></details>","Drop","zstd_safe::CStream"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()