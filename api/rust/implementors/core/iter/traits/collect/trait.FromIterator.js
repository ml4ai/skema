(function() {var implementors = {
"ahash":[["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"ahash/struct.AHashSet.html\" title=\"struct ahash::AHashSet\">AHashSet</a>&lt;T, <a class=\"struct\" href=\"ahash/random_state/struct.RandomState.html\" title=\"struct ahash::random_state::RandomState\">RandomState</a>&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"],["impl&lt;K, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"ahash/struct.AHashMap.html\" title=\"struct ahash::AHashMap\">AHashMap</a>&lt;K, V, <a class=\"struct\" href=\"ahash/random_state/struct.RandomState.html\" title=\"struct ahash::random_state::RandomState\">RandomState</a>&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"]],
"bytes":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.u8.html\">u8</a>&gt; for <a class=\"struct\" href=\"bytes/struct.BytesMut.html\" title=\"struct bytes::BytesMut\">BytesMut</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.u8.html\">u8</a>&gt; for <a class=\"struct\" href=\"bytes/struct.Bytes.html\" title=\"struct bytes::Bytes\">Bytes</a>"],["impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;&amp;'a <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.u8.html\">u8</a>&gt; for <a class=\"struct\" href=\"bytes/struct.BytesMut.html\" title=\"struct bytes::BytesMut\">BytesMut</a>"]],
"fixedbitset":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/core/primitive.usize.html\">usize</a>&gt; for <a class=\"struct\" href=\"fixedbitset/struct.FixedBitSet.html\" title=\"struct fixedbitset::FixedBitSet\">FixedBitSet</a>"]],
"futures_util":[["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;F&gt; for <a class=\"struct\" href=\"futures_util/future/struct.JoinAll.html\" title=\"struct futures_util::future::JoinAll\">JoinAll</a>&lt;F&gt;"],["impl&lt;F&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;F&gt; for <a class=\"struct\" href=\"futures_util/future/struct.TryJoinAll.html\" title=\"struct futures_util::future::TryJoinAll\">TryJoinAll</a>&lt;F&gt;<span class=\"where fmt-newline\">where\n    F: <a class=\"trait\" href=\"futures_util/future/trait.TryFuture.html\" title=\"trait futures_util::future::TryFuture\">TryFuture</a>,</span>"],["impl&lt;Fut: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;Fut&gt; for <a class=\"struct\" href=\"futures_util/future/struct.SelectAll.html\" title=\"struct futures_util::future::SelectAll\">SelectAll</a>&lt;Fut&gt;"],["impl&lt;Fut&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;Fut&gt; for <a class=\"struct\" href=\"futures_util/stream/struct.FuturesUnordered.html\" title=\"struct futures_util::stream::FuturesUnordered\">FuturesUnordered</a>&lt;Fut&gt;"],["impl&lt;St: <a class=\"trait\" href=\"futures_util/stream/trait.Stream.html\" title=\"trait futures_util::stream::Stream\">Stream</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;St&gt; for <a class=\"struct\" href=\"futures_util/stream/struct.SelectAll.html\" title=\"struct futures_util::stream::SelectAll\">SelectAll</a>&lt;St&gt;"],["impl&lt;Fut: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;Fut&gt; for <a class=\"struct\" href=\"futures_util/stream/struct.FuturesOrdered.html\" title=\"struct futures_util::stream::FuturesOrdered\">FuturesOrdered</a>&lt;Fut&gt;"],["impl&lt;Fut: <a class=\"trait\" href=\"futures_util/future/trait.TryFuture.html\" title=\"trait futures_util::future::TryFuture\">TryFuture</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;Fut&gt; for <a class=\"struct\" href=\"futures_util/future/struct.SelectOk.html\" title=\"struct futures_util::future::SelectOk\">SelectOk</a>&lt;Fut&gt;"]],
"generic_array":[["impl&lt;T, N&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"generic_array/struct.GenericArray.html\" title=\"struct generic_array::GenericArray\">GenericArray</a>&lt;T, N&gt;<span class=\"where fmt-newline\">where\n    N: <a class=\"trait\" href=\"generic_array/trait.ArrayLength.html\" title=\"trait generic_array::ArrayLength\">ArrayLength</a>&lt;T&gt;,</span>"]],
"hashbrown":[["impl&lt;T, S, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"hashbrown/hash_set/struct.HashSet.html\" title=\"struct hashbrown::hash_set::HashSet\">HashSet</a>&lt;T, S, A&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    A: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> + Allocator + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</span>"],["impl&lt;K, V, S, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/core/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"hashbrown/hash_map/struct.HashMap.html\" title=\"struct hashbrown::hash_map::HashMap\">HashMap</a>&lt;K, V, S, A&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,\n    A: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> + Allocator + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</span>"]],
"http":[["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"struct\" href=\"http/header/struct.HeaderName.html\" title=\"struct http::header::HeaderName\">HeaderName</a>, T)&gt; for <a class=\"struct\" href=\"http/header/struct.HeaderMap.html\" title=\"struct http::header::HeaderMap\">HeaderMap</a>&lt;T&gt;"]],
"indexmap":[["impl&lt;K, V, S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"struct\" href=\"indexmap/map/struct.IndexMap.html\" title=\"struct indexmap::map::IndexMap\">IndexMap</a>&lt;K, V, S&gt;<span class=\"where fmt-newline\">where\n    K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"],["impl&lt;T, S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"indexmap/set/struct.IndexSet.html\" title=\"struct indexmap::set::IndexSet\">IndexSet</a>&lt;T, S&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>,\n    S: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/hash/trait.BuildHasher.html\" title=\"trait core::hash::BuildHasher\">BuildHasher</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"]],
"password_hash":[["impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"struct\" href=\"password_hash/struct.Ident.html\" title=\"struct password_hash::Ident\">Ident</a>&lt;'a&gt;, <a class=\"struct\" href=\"password_hash/struct.Value.html\" title=\"struct password_hash::Value\">Value</a>&lt;'a&gt;)&gt; for <a class=\"struct\" href=\"password_hash/struct.ParamsString.html\" title=\"struct password_hash::ParamsString\">ParamsString</a>"]],
"petgraph":[["impl&lt;N, E, Ty, Item&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;Item&gt; for <a class=\"struct\" href=\"petgraph/graphmap/struct.GraphMap.html\" title=\"struct petgraph::graphmap::GraphMap\">GraphMap</a>&lt;N, E, Ty&gt;<span class=\"where fmt-newline\">where\n    Item: <a class=\"trait\" href=\"petgraph/trait.IntoWeightedEdge.html\" title=\"trait petgraph::IntoWeightedEdge\">IntoWeightedEdge</a>&lt;E, NodeId = N&gt;,\n    N: <a class=\"trait\" href=\"petgraph/graphmap/trait.NodeTrait.html\" title=\"trait petgraph::graphmap::NodeTrait\">NodeTrait</a>,\n    Ty: <a class=\"trait\" href=\"petgraph/trait.EdgeType.html\" title=\"trait petgraph::EdgeType\">EdgeType</a>,</span>"]],
"proc_macro2":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>&gt; for <a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"enum\" href=\"proc_macro2/enum.TokenTree.html\" title=\"enum proc_macro2::TokenTree\">TokenTree</a>&gt; for <a class=\"struct\" href=\"proc_macro2/struct.TokenStream.html\" title=\"struct proc_macro2::TokenStream\">TokenStream</a>"]],
"rustix":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PollFlags.html\" title=\"struct rustix::io::PollFlags\">PollFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.FdFlags.html\" title=\"struct rustix::io::FdFlags\">FdFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.PipeFlags.html\" title=\"struct rustix::io::PipeFlags\">PipeFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.EventfdFlags.html\" title=\"struct rustix::io::EventfdFlags\">EventfdFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.DupFlags.html\" title=\"struct rustix::io::DupFlags\">DupFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.CreateFlags.html\" title=\"struct rustix::io::epoll::CreateFlags\">CreateFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.SpliceFlags.html\" title=\"struct rustix::io::SpliceFlags\">SpliceFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/epoll/struct.EventFlags.html\" title=\"struct rustix::io::epoll::EventFlags\">EventFlags</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>&gt; for <a class=\"struct\" href=\"rustix/io/struct.ReadWriteFlags.html\" title=\"struct rustix::io::ReadWriteFlags\">ReadWriteFlags</a>"]],
"serde_json":[["impl&lt;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.70.0/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt;, V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.tuple.html\">(K, V)</a>&gt; for <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"struct\" href=\"https://doc.rust-lang.org/1.70.0/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>, <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>)&gt; for <a class=\"struct\" href=\"serde_json/struct.Map.html\" title=\"struct serde_json::Map\">Map</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.70.0/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>, <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;"],["impl&lt;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"enum\" href=\"serde_json/enum.Value.html\" title=\"enum serde_json::Value\">Value</a>"]],
"serde_yaml":[["impl&lt;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"serde_yaml/enum.Value.html\" title=\"enum serde_yaml::Value\">Value</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"enum\" href=\"serde_yaml/enum.Value.html\" title=\"enum serde_yaml::Value\">Value</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"enum\" href=\"serde_yaml/enum.Value.html\" title=\"enum serde_yaml::Value\">Value</a>, <a class=\"enum\" href=\"serde_yaml/enum.Value.html\" title=\"enum serde_yaml::Value\">Value</a>)&gt; for <a class=\"struct\" href=\"serde_yaml/struct.Mapping.html\" title=\"struct serde_yaml::Mapping\">Mapping</a>"]],
"slab":[["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;(<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.usize.html\">usize</a>, T)&gt; for <a class=\"struct\" href=\"slab/struct.Slab.html\" title=\"struct slab::Slab\">Slab</a>&lt;T&gt;"]],
"smallvec":[["impl&lt;A: <a class=\"trait\" href=\"smallvec/trait.Array.html\" title=\"trait smallvec::Array\">Array</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;&lt;A as <a class=\"trait\" href=\"smallvec/trait.Array.html\" title=\"trait smallvec::Array\">Array</a>&gt;::<a class=\"associatedtype\" href=\"smallvec/trait.Array.html#associatedtype.Item\" title=\"type smallvec::Array::Item\">Item</a>&gt; for <a class=\"struct\" href=\"smallvec/struct.SmallVec.html\" title=\"struct smallvec::SmallVec\">SmallVec</a>&lt;A&gt;"]],
"syn":[["impl&lt;T, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;T&gt; for <a class=\"struct\" href=\"syn/punctuated/struct.Punctuated.html\" title=\"struct syn::punctuated::Punctuated\">Punctuated</a>&lt;T, P&gt;<span class=\"where fmt-newline\">where\n    P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a>,</span>"],["impl&lt;T, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"enum\" href=\"syn/punctuated/enum.Pair.html\" title=\"enum syn::punctuated::Pair\">Pair</a>&lt;T, P&gt;&gt; for <a class=\"struct\" href=\"syn/punctuated/struct.Punctuated.html\" title=\"struct syn::punctuated::Punctuated\">Punctuated</a>&lt;T, P&gt;"]],
"tinyvec":[["impl&lt;A: <a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;&lt;A as <a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt;::<a class=\"associatedtype\" href=\"tinyvec/trait.Array.html#associatedtype.Item\" title=\"type tinyvec::Array::Item\">Item</a>&gt; for <a class=\"struct\" href=\"tinyvec/struct.ArrayVec.html\" title=\"struct tinyvec::ArrayVec\">ArrayVec</a>&lt;A&gt;"],["impl&lt;A: <a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;&lt;A as <a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt;::<a class=\"associatedtype\" href=\"tinyvec/trait.Array.html#associatedtype.Item\" title=\"type tinyvec::Array::Item\">Item</a>&gt; for <a class=\"enum\" href=\"tinyvec/enum.TinyVec.html\" title=\"enum tinyvec::TinyVec\">TinyVec</a>&lt;A&gt;"]],
"utoipa":[["impl&lt;C, R&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.tuple.html\">(C, R)</a>&gt; for <a class=\"struct\" href=\"utoipa/openapi/response/struct.Responses.html\" title=\"struct utoipa::openapi::response::Responses\">Responses</a><span class=\"where fmt-newline\">where\n    C: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.70.0/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt;,\n    R: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"enum\" href=\"utoipa/openapi/enum.RefOr.html\" title=\"enum utoipa::openapi::RefOr\">RefOr</a>&lt;<a class=\"struct\" href=\"utoipa/openapi/response/struct.Response.html\" title=\"struct utoipa::openapi::response::Response\">Response</a>&gt;&gt;,</span>"],["impl&lt;I&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/iter/traits/collect/trait.FromIterator.html\" title=\"trait core::iter::traits::collect::FromIterator\">FromIterator</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.70.0/std/primitive.tuple.html\">(I, I)</a>&gt; for <a class=\"struct\" href=\"utoipa/openapi/security/struct.Scopes.html\" title=\"struct utoipa::openapi::security::Scopes\">Scopes</a><span class=\"where fmt-newline\">where\n    I: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.70.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.70.0/alloc/string/struct.String.html\" title=\"struct alloc::string::String\">String</a>&gt;,</span>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()