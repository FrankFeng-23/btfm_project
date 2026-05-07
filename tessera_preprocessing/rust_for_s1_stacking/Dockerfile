# Use Alpine Linux to build static linked binary
FROM rust:1.70-alpine as builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
ENV RUSTFLAGS="-C target-feature=+crt-static"

# Install necessary build dependencies
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    openssl-libs-static \
    gcc \
    g++ \
    make

# First copy Cargo.toml and Cargo.lock to leverage build cache
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main.rs file to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --target x86_64-unknown-linux-musl && \
    rm -f target/x86_64-unknown-linux-musl/release/deps/s1_stack*

# Copy source code
COPY . .

# Build application (statically linked)
RUN cargo build --release --target x86_64-unknown-linux-musl

# Use scratch as final image (since it's statically linked, no runtime libraries needed)
FROM scratch

# Copy binary from build stage
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/s1_stack /s1_stack

# Set entry point
ENTRYPOINT ["/s1_stack"]

# Build instructions:
# docker build -t s1_stack .
# docker create --name temp_container s1_stack
# docker cp temp_container:/s1_stack ./s1_stack
# docker rm temp_container
# 
# Verify static linking:
# ldd ./s1_stack
# Should show "not a dynamic executable"