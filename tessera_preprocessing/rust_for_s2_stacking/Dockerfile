# Use Alpine Linux to build a statically linked binary
FROM rust:1.83-alpine as builder

# Install required build tools
RUN apk add --no-cache musl-dev

# Set working directory
WORKDIR /app

# Copy Cargo.toml
COPY Cargo.toml ./

# Create src directory and copy source code
RUN mkdir -p src
COPY src/main.rs ./src/main.rs

# Configure Rust to target musl
ENV RUSTFLAGS="-C target-feature=+crt-static"

# Build the release binary (statically linked)
RUN cargo build --release --target x86_64-unknown-linux-musl

# Second stage: use scratch (minimal image)
FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/process_tile_downstream_wo_json /s2_stack
ENTRYPOINT ["/s2_stack"]

# docker build -t s2_stack .
# docker create --name temp_container s2_stack
# docker cp temp_container:/s2_stack ./s2_stack
# docker rm temp_container
