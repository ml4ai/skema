(function() {var implementors = {
"actix_http":[["impl&lt;T, S, B, X, U&gt; <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(T, <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/net/socket_addr/enum.SocketAddr.html\" title=\"enum core::net::socket_addr::SocketAddr\">SocketAddr</a>&gt;)&gt; for <a class=\"struct\" href=\"actix_http/h1/struct.H1Service.html\" title=\"struct actix_http::h1::H1Service\">H1Service</a>&lt;T, S, B, X, U&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> + <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a> + 'static,\n    S: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>&gt;,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt;,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Response\" title=\"type actix_service::ServiceFactory::Response\">Response</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;B&gt;&gt;,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,\n    B: <a class=\"trait\" href=\"actix_http/body/trait.MessageBody.html\" title=\"trait actix_http::body::MessageBody\">MessageBody</a>,\n    X: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>, Response = <a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>&gt;,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt;,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,\n    U: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, <a class=\"struct\" href=\"actix_codec/framed/struct.Framed.html\" title=\"struct actix_codec::framed::Framed\">Framed</a>&lt;T, <a class=\"struct\" href=\"actix_http/h1/struct.Codec.html\" title=\"struct actix_http::h1::Codec\">Codec</a>&gt;), Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>, Response = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>&gt;,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Display.html\" title=\"trait core::fmt::Display\">Display</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt;,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,</span>"],["impl&lt;T, S, B, X, U&gt; <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(T, <a class=\"enum\" href=\"actix_http/enum.Protocol.html\" title=\"enum actix_http::Protocol\">Protocol</a>, <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/net/socket_addr/enum.SocketAddr.html\" title=\"enum core::net::socket_addr::SocketAddr\">SocketAddr</a>&gt;)&gt; for <a class=\"struct\" href=\"actix_http/struct.HttpService.html\" title=\"struct actix_http::HttpService\">HttpService</a>&lt;T, S, B, X, U&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> + <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a> + 'static,\n    S: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>&gt;,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt; + 'static,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Response\" title=\"type actix_service::ServiceFactory::Response\">Response</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;B&gt;&gt; + 'static,\n    &lt;S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Service\" title=\"type actix_service::ServiceFactory::Service\">Service</a> as <a class=\"trait\" href=\"actix_service/trait.Service.html\" title=\"trait actix_service::Service\">Service</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>&gt;&gt;::<a class=\"associatedtype\" href=\"actix_service/trait.Service.html#associatedtype.Future\" title=\"type actix_service::Service::Future\">Future</a>: 'static,\n    B: <a class=\"trait\" href=\"actix_http/body/trait.MessageBody.html\" title=\"trait actix_http::body::MessageBody\">MessageBody</a> + 'static,\n    X: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>, Response = <a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>&gt;,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt;,\n    X::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,\n    U: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, <a class=\"struct\" href=\"actix_codec/framed/struct.Framed.html\" title=\"struct actix_codec::framed::Framed\">Framed</a>&lt;T, <a class=\"struct\" href=\"actix_http/h1/struct.Codec.html\" title=\"struct actix_http::h1::Codec\">Codec</a>&gt;), Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>, Response = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>&gt;,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Display.html\" title=\"trait core::fmt::Display\">Display</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt;,\n    U::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.InitError\" title=\"type actix_service::ServiceFactory::InitError\">InitError</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,</span>"],["impl&lt;T, S, B&gt; <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(T, <a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.75.0/core/net/socket_addr/enum.SocketAddr.html\" title=\"enum core::net::socket_addr::SocketAddr\">SocketAddr</a>&gt;)&gt; for <a class=\"struct\" href=\"actix_http/h2/struct.H2Service.html\" title=\"struct actix_http::h2::H2Service\">H2Service</a>&lt;T, S, B&gt;<span class=\"where fmt-newline\">where\n    T: <a class=\"trait\" href=\"tokio/io/async_read/trait.AsyncRead.html\" title=\"trait tokio::io::async_read::AsyncRead\">AsyncRead</a> + <a class=\"trait\" href=\"tokio/io/async_write/trait.AsyncWrite.html\" title=\"trait tokio::io::async_write::AsyncWrite\">AsyncWrite</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a> + 'static,\n    S: <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, Config = <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.75.0/std/primitive.unit.html\">()</a>&gt;,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Future\" title=\"type actix_service::ServiceFactory::Future\">Future</a>: 'static,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Error\" title=\"type actix_service::ServiceFactory::Error\">Error</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;<a class=\"struct\" href=\"actix_http/body/struct.BoxBody.html\" title=\"struct actix_http::body::BoxBody\">BoxBody</a>&gt;&gt; + 'static,\n    S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Response\" title=\"type actix_service::ServiceFactory::Response\">Response</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.75.0/core/convert/trait.Into.html\" title=\"trait core::convert::Into\">Into</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Response.html\" title=\"struct actix_http::Response\">Response</a>&lt;B&gt;&gt; + 'static,\n    &lt;S::<a class=\"associatedtype\" href=\"actix_service/trait.ServiceFactory.html#associatedtype.Service\" title=\"type actix_service::ServiceFactory::Service\">Service</a> as <a class=\"trait\" href=\"actix_service/trait.Service.html\" title=\"trait actix_service::Service\">Service</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>&gt;&gt;::<a class=\"associatedtype\" href=\"actix_service/trait.Service.html#associatedtype.Future\" title=\"type actix_service::Service::Future\">Future</a>: 'static,\n    B: <a class=\"trait\" href=\"actix_http/body/trait.MessageBody.html\" title=\"trait actix_http::body::MessageBody\">MessageBody</a> + 'static,</span>"],["impl <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>&gt; for <a class=\"struct\" href=\"actix_http/h1/struct.ExpectHandler.html\" title=\"struct actix_http::h1::ExpectHandler\">ExpectHandler</a>"],["impl&lt;T&gt; <a class=\"trait\" href=\"actix_service/trait.ServiceFactory.html\" title=\"trait actix_service::ServiceFactory\">ServiceFactory</a>&lt;(<a class=\"struct\" href=\"actix_http/struct.Request.html\" title=\"struct actix_http::Request\">Request</a>, <a class=\"struct\" href=\"actix_codec/framed/struct.Framed.html\" title=\"struct actix_codec::framed::Framed\">Framed</a>&lt;T, <a class=\"struct\" href=\"actix_http/h1/struct.Codec.html\" title=\"struct actix_http::h1::Codec\">Codec</a>&gt;)&gt; for <a class=\"struct\" href=\"actix_http/h1/struct.UpgradeHandler.html\" title=\"struct actix_http::h1::UpgradeHandler\">UpgradeHandler</a>"]],
"actix_service":[],
"actix_web":[["impl <a class=\"trait\" href=\"actix_web/dev/trait.ServiceFactory.html\" title=\"trait actix_web::dev::ServiceFactory\">ServiceFactory</a>&lt;<a class=\"struct\" href=\"actix_web/dev/struct.ServiceRequest.html\" title=\"struct actix_web::dev::ServiceRequest\">ServiceRequest</a>&gt; for <a class=\"struct\" href=\"actix_web/struct.Route.html\" title=\"struct actix_web::Route\">Route</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()