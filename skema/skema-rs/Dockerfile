from rust:1.65

run apt update \
    && apt install -y \
        cmake openssl libclang-13-dev
workdir /usr/src/myapp
copy . .
run cargo build --release --bin skema_service
