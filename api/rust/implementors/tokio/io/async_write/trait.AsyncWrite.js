(function() {var implementors = {
"actix_codec":[],
"actix_http":[["impl <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"struct\" href=\"actix_http/test/struct.TestBuffer.html\" title=\"struct actix_http::test::TestBuffer\">TestBuffer</a>"],["impl <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"struct\" href=\"actix_http/test/struct.TestSeqBuffer.html\" title=\"struct actix_http::test::TestSeqBuffer\">TestSeqBuffer</a>"]],
"tokio":[],
"tokio_util":[["impl&lt;R: <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a>, F&gt; <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"struct\" href=\"tokio_util/io/struct.InspectReader.html\" title=\"struct tokio_util::io::InspectReader\">InspectReader</a>&lt;R, F&gt;"],["impl&lt;W: <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a>, F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.72.1/core/ops/function/trait.FnMut.html\" title=\"trait core::ops::function::FnMut\">FnMut</a>(&amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.72.1/std/primitive.u8.html\">u8</a>])&gt; <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"struct\" href=\"tokio_util/io/struct.InspectWriter.html\" title=\"struct tokio_util::io::InspectWriter\">InspectWriter</a>&lt;W, F&gt;"],["impl&lt;L, R&gt; <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"enum\" href=\"tokio_util/either/enum.Either.html\" title=\"enum tokio_util::either::Either\">Either</a>&lt;L, R&gt;<span class=\"where fmt-newline\">where\n    L: <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a>,\n    R: <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a>,</span>"],["impl&lt;S, E&gt; <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> for <a class=\"struct\" href=\"tokio_util/io/struct.SinkWriter.html\" title=\"struct tokio_util::io::SinkWriter\">SinkWriter</a>&lt;S&gt;<span class=\"where fmt-newline\">where\n    for&lt;'a&gt; S: <a class=\"trait\" href=\"futures_sink/trait.Sink.html\" title=\"trait futures_sink::Sink\">Sink</a>&lt;&amp;'a [<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.72.1/std/primitive.u8.html\">u8</a>], Error = E&gt;,\n    E: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.72.1/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.72.1/std/io/error/struct.Error.html\" title=\"struct std::io::error::Error\">Error</a>&gt;,</span>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()