(function() {var implementors = {
"actix_codec":[],
"actix_http":[["impl <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"actix_http/test/struct.TestBuffer.html\" title=\"struct actix_http::test::TestBuffer\">TestBuffer</a>"],["impl <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"actix_http/test/struct.TestSeqBuffer.html\" title=\"struct actix_http::test::TestSeqBuffer\">TestSeqBuffer</a>"]],
"tokio":[],
"tokio_util":[["impl&lt;S, B, E&gt; <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"tokio_util/io/struct.StreamReader.html\" title=\"struct tokio_util::io::StreamReader\">StreamReader</a>&lt;S, B&gt;<span class=\"where fmt-newline\">where\n    S: <a class=\"trait\" href=\"futures_core/stream/trait.Stream.html\" title=\"trait futures_core::stream::Stream\">Stream</a>&lt;Item = <a class=\"enum\" href=\"https://doc.rust-lang.org/1.73.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;B, E&gt;&gt;,\n    B: <a class=\"trait\" href=\"bytes/buf/buf_impl/trait.Buf.html\" title=\"trait bytes::buf::buf_impl::Buf\">Buf</a>,\n    E: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.73.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.73.0/std/io/error/struct.Error.html\" title=\"struct std::io::error::Error\">Error</a>&gt;,</span>"],["impl&lt;W: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a>, F&gt; <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"tokio_util/io/struct.InspectWriter.html\" title=\"struct tokio_util::io::InspectWriter\">InspectWriter</a>&lt;W, F&gt;"],["impl&lt;S: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a>&gt; <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"tokio_util/io/struct.SinkWriter.html\" title=\"struct tokio_util::io::SinkWriter\">SinkWriter</a>&lt;S&gt;"],["impl&lt;R: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a>, F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.73.0/core/ops/function/trait.FnMut.html\" title=\"trait core::ops::function::FnMut\">FnMut</a>(&amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.73.0/std/primitive.u8.html\">u8</a>])&gt; <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"struct\" href=\"tokio_util/io/struct.InspectReader.html\" title=\"struct tokio_util::io::InspectReader\">InspectReader</a>&lt;R, F&gt;"],["impl&lt;L, R&gt; <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> for <a class=\"enum\" href=\"tokio_util/either/enum.Either.html\" title=\"enum tokio_util::either::Either\">Either</a>&lt;L, R&gt;<span class=\"where fmt-newline\">where\n    L: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a>,\n    R: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a>,</span>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()