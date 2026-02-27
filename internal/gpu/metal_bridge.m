//go:build darwin

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include "metal_bridge.h"

static NSUInteger bestThreadGroupSize(NSUInteger maxThreadsPerGroup, NSUInteger threadExecutionWidth) {
    if (maxThreadsPerGroup == 0) return 1;
    NSUInteger groupSize = maxThreadsPerGroup;
    if (groupSize > 256) groupSize = 256;
    if (threadExecutionWidth > 0) {
        groupSize = (groupSize / threadExecutionWidth) * threadExecutionWidth;
        if (groupSize == 0) groupSize = threadExecutionWidth;
    }
    return groupSize > 0 ? groupSize : 1;
}

static inline uint32_t rotr32_host(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t sig0_host(uint32_t x) {
    return rotr32_host(x, 7) ^ rotr32_host(x, 18) ^ (x >> 3);
}

static inline uint32_t sig1_host(uint32_t x) {
    return rotr32_host(x, 17) ^ rotr32_host(x, 19) ^ (x >> 10);
}

static inline uint32_t readBE32Host(const unsigned char* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Embedded Metal shader source (SHA-256 + base32 prefix check)
static NSString* const shaderSource = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"// SHA-256 round constants\n"
"constant uint K[64] = {\n"
"    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,\n"
"    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,\n"
"    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,\n"
"    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,\n"
"    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,\n"
"    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,\n"
"    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,\n"
"    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,\n"
"    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,\n"
"    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,\n"
"    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,\n"
"    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,\n"
"    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,\n"
"    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,\n"
"    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,\n"
"    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2\n"
"};\n"
"\n"
"// SHA-256 helper functions\n"
"inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }\n"
"inline uint ch(uint x, uint y, uint z) { return (x & y) ^ (~x & z); }\n"
"inline uint maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }\n"
"inline uint ep0(uint x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }\n"
"inline uint ep1(uint x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }\n"
"inline uint sig0(uint x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }\n"
"inline uint sig1(uint x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }\n"
"\n"
"// Base32 alphabet\n"
"constant char B32[32] = {\n"
"    'a','b','c','d','e','f','g','h','i','j','k','l','m',\n"
"    'n','o','p','q','r','s','t','u','v','w','x','y','z',\n"
"    '2','3','4','5','6','7'\n"
"};\n"
"\n"
"struct VanityParams {\n"
"    ulong counter_base;\n"
"    uint prefix_len;\n"
"    char prefix[64];\n"
"    uint block0_words[14];\n"
"};\n"
"\n"
"kernel void vanity_search(\n"
"    constant VanityParams* params [[buffer(0)]],\n"
"    constant uint* static_w [[buffer(1)]],\n"
"    device atomic_int* match_found [[buffer(2)]],\n"
"    device ulong* match_counter [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    ulong counter = params->counter_base + (ulong)gid;\n"
"    \n"
"    // Early exit if another thread already found a match\n"
"    if (atomic_load_explicit(match_found, memory_order_relaxed) != 0) return;\n"
"    \n"
"    // SHA-256 compression\n"
"    uint h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;\n"
"    uint h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;\n"
"    \n"
"    // Block 0 has dynamic counter bytes and static words from params\n"
"    uint w[64];\n"
"    uint c0 = (uint)(counter & 0xFFFFFFFFUL);\n"
"    uint c1 = (uint)(counter >> 32);\n"
"    w[0] = ((c0 & 0x000000FFu) << 24) | ((c0 & 0x0000FF00u) << 8) |\n"
"           ((c0 & 0x00FF0000u) >> 8)  | ((c0 & 0xFF000000u) >> 24);\n"
"    w[1] = ((c1 & 0x000000FFu) << 24) | ((c1 & 0x0000FF00u) << 8) |\n"
"           ((c1 & 0x00FF0000u) >> 8)  | ((c1 & 0xFF000000u) >> 24);\n"
"    for (uint i = 0; i < 14; i++) {\n"
"        w[i + 2] = params->block0_words[i];\n"
"    }\n"
"    for (uint i = 16; i < 64; i++) {\n"
"        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];\n"
"    }\n"
"    \n"
"    uint a = h0, b = h1, c = h2, d = h3;\n"
"    uint e = h4, f = h5, g = h6, h = h7;\n"
"    for (uint i = 0; i < 64; i++) {\n"
"        uint t1 = h + ep1(e) + ch(e, f, g) + K[i] + w[i];\n"
"        uint t2 = ep0(a) + maj(a, b, c);\n"
"        h = g; g = f; f = e; e = d + t1;\n"
"        d = c; c = b; b = a; a = t1 + t2;\n"
"    }\n"
"    h0 += a; h1 += b; h2 += c; h3 += d;\n"
"    h4 += e; h5 += f; h6 += g; h7 += h;\n"
"    \n"
"    // Blocks 1..6 use pre-expanded static schedules from static_w\n"
"    for (uint block = 0; block < 6; block++) {\n"
"        constant uint* ws = static_w + block * 64;\n"
"        a = h0; b = h1; c = h2; d = h3;\n"
"        e = h4; f = h5; g = h6; h = h7;\n"
"        for (uint i = 0; i < 64; i++) {\n"
"            uint t1 = h + ep1(e) + ch(e, f, g) + K[i] + ws[i];\n"
"            uint t2 = ep0(a) + maj(a, b, c);\n"
"            h = g; g = f; f = e; e = d + t1;\n"
"            d = c; c = b; b = a; a = t1 + t2;\n"
"        }\n"
"        h0 += a; h1 += b; h2 += c; h3 += d;\n"
"        h4 += e; h5 += f; h6 += g; h7 += h;\n"
"    }\n"
"    \n"
"    // Extract hash bytes (only need enough for prefix check)\n"
"    uchar hash[33];\n"
"    hash[0] = (h0 >> 24); hash[1] = (h0 >> 16); hash[2] = (h0 >> 8); hash[3] = h0;\n"
"    hash[4] = (h1 >> 24); hash[5] = (h1 >> 16); hash[6] = (h1 >> 8); hash[7] = h1;\n"
"    hash[8] = (h2 >> 24); hash[9] = (h2 >> 16); hash[10] = (h2 >> 8); hash[11] = h2;\n"
"    hash[12] = (h3 >> 24); hash[13] = (h3 >> 16); hash[14] = (h3 >> 8); hash[15] = h3;\n"
"    hash[16] = (h4 >> 24); hash[17] = (h4 >> 16); hash[18] = (h4 >> 8); hash[19] = h4;\n"
"    hash[20] = (h5 >> 24); hash[21] = (h5 >> 16); hash[22] = (h5 >> 8); hash[23] = h5;\n"
"    hash[24] = (h6 >> 24); hash[25] = (h6 >> 16); hash[26] = (h6 >> 8); hash[27] = h6;\n"
"    hash[28] = (h7 >> 24); hash[29] = (h7 >> 16); hash[30] = (h7 >> 8); hash[31] = h7;\n"
"    hash[32] = 0;\n"
"    \n"
"    // Base32 encode and compare prefix\n"
"    uint prefix_len = params->prefix_len;\n"
"    uint bit_offset = 0;\n"
"    bool match = true;\n"
"    for (uint i = 0; i < prefix_len && match; i++) {\n"
"        uint byte_idx = bit_offset / 8;\n"
"        uint bit_idx = bit_offset % 8;\n"
"        uint val;\n"
"        if (bit_idx <= 3) {\n"
"            val = (hash[byte_idx] >> (3 - bit_idx)) & 0x1f;\n"
"        } else {\n"
"            val = ((hash[byte_idx] << (bit_idx - 3)) | (hash[byte_idx + 1] >> (11 - bit_idx))) & 0x1f;\n"
"        }\n"
"        if (B32[val] != params->prefix[i]) {\n"
"            match = false;\n"
"        }\n"
"        bit_offset += 5;\n"
"    }\n"
"    \n"
"    if (match) {\n"
"        // Atomically signal match (only first match wins)\n"
"        int expected = 0;\n"
"        if (atomic_compare_exchange_weak_explicit(match_found, &expected, 1,\n"
"                memory_order_relaxed, memory_order_relaxed)) {\n"
"            *match_counter = counter;\n"
"        }\n"
"    }\n"
"}\n";

// ---- Bridge implementation ----

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> staticWBuf;
    id<MTLBuffer> paramsBuf;
    id<MTLBuffer> matchFoundBuf;
    id<MTLBuffer> matchCounterBuf;
    NSUInteger batchSize;
    NSUInteger maxThreadsPerGroup;
    NSUInteger threadExecutionWidth;
} MetalWorker;

// Packed to match shader struct
typedef struct __attribute__((packed)) {
    uint64_t counter_base;
    uint32_t prefix_len;
    char prefix[64];
    uint32_t block0_words[14];
} VanityParams;

int metalAvailable(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil ? 1 : 0;
    }
}

char** metalListDevices(int* count) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices == nil || [devices count] == 0) {
            // Fall back to default device
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (dev == nil) {
                *count = 0;
                return NULL;
            }
            *count = 1;
            char** names = (char**)malloc(sizeof(char*));
            const char* name = [[dev name] UTF8String];
            names[0] = strdup(name);
            return names;
        }

        int n = (int)[devices count];
        *count = n;
        char** names = (char**)malloc(sizeof(char*) * n);
        for (int i = 0; i < n; i++) {
            const char* name = [[devices[i] name] UTF8String];
            names[i] = strdup(name);
        }
        return names;
    }
}

void* metalNewWorker(int deviceIndex, const unsigned char* destTemplate,
                     const char* prefix, int prefixLen, unsigned long batchSize) {
    @autoreleasepool {
        if (batchSize == 0) return NULL;

        // Get device
        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && [devices count] > 0 && deviceIndex < (int)[devices count]) {
            device = devices[deviceIndex];
        } else {
            device = MTLCreateSystemDefaultDevice();
        }
        if (device == nil) return NULL;

        // Compile shader
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:options error:&error];
        if (library == nil) {
            NSLog(@"Metal shader compile error: %@", error);
            return NULL;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"vanity_search"];
        if (function == nil) {
            NSLog(@"Metal function 'vanity_search' not found");
            return NULL;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            NSLog(@"Metal pipeline error: %@", error);
            return NULL;
        }

        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return NULL;

        VanityParams params;
        memset(&params, 0, sizeof(params));
        params.counter_base = 0;
        params.prefix_len = (uint32_t)prefixLen;
        size_t prefixCopyLen = (size_t)(prefixLen < 64 ? prefixLen : 64);
        memcpy(params.prefix, prefix, prefixCopyLen);
        for (int i = 0; i < 14; i++) {
            params.block0_words[i] = readBE32Host(destTemplate + 8 + i*4);
        }

        uint32_t staticW[6][64];
        for (int b = 0; b < 6; b++) {
            unsigned char block[64];
            memset(block, 0, sizeof(block));
            if (b < 5) {
                memcpy(block, destTemplate + (size_t)(b+1)*64, 64);
            } else {
                memcpy(block, destTemplate + 384, 7);
                block[7] = 0x80;
                block[62] = 0x0C;
                block[63] = 0x38;
            }
            for (int i = 0; i < 16; i++) {
                staticW[b][i] = readBE32Host(block + i*4);
            }
            for (int i = 16; i < 64; i++) {
                staticW[b][i] = sig1_host(staticW[b][i-2]) + staticW[b][i-7] + sig0_host(staticW[b][i-15]) + staticW[b][i-16];
            }
        }

        id<MTLBuffer> staticWBuf = [device newBufferWithBytes:staticW
                                                       length:sizeof(staticW)
                                                      options:MTLResourceStorageModeShared];
        if (staticWBuf == nil) return NULL;

        id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                      length:sizeof(VanityParams)
                                                     options:MTLResourceStorageModeShared];

        int32_t zero = 0;
        id<MTLBuffer> matchFoundBuf = [device newBufferWithBytes:&zero
                                                          length:sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared];

        uint64_t zeroCounter = 0;
        id<MTLBuffer> matchCounterBuf = [device newBufferWithBytes:&zeroCounter
                                                            length:sizeof(uint64_t)
                                                           options:MTLResourceStorageModeShared];

        // Allocate worker struct
        MetalWorker* worker = (MetalWorker*)calloc(1, sizeof(MetalWorker));
        worker->device = device;
        worker->queue = queue;
        worker->pipeline = pipeline;
        worker->staticWBuf = staticWBuf;
        worker->paramsBuf = paramsBuf;
        worker->matchFoundBuf = matchFoundBuf;
        worker->matchCounterBuf = matchCounterBuf;
        worker->batchSize = (NSUInteger)batchSize;
        worker->maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
        worker->threadExecutionWidth = [pipeline threadExecutionWidth];

        // Retain Objective-C objects
        CFRetain((__bridge CFTypeRef)device);
        CFRetain((__bridge CFTypeRef)queue);
        CFRetain((__bridge CFTypeRef)pipeline);
        CFRetain((__bridge CFTypeRef)staticWBuf);
        CFRetain((__bridge CFTypeRef)paramsBuf);
        CFRetain((__bridge CFTypeRef)matchFoundBuf);
        CFRetain((__bridge CFTypeRef)matchCounterBuf);

        return worker;
    }
}

unsigned long metalRunBatch(void* handle, unsigned long counterStart,
                            int* matchFound, unsigned long* matchCounter) {
    @autoreleasepool {
        MetalWorker* worker = (MetalWorker*)handle;
        if (!worker) return 0;

        // Update params with new counter_base
        VanityParams* params = (VanityParams*)[worker->paramsBuf contents];
        params->counter_base = (uint64_t)counterStart;

        // Reset match_found to 0
        int32_t* found = (int32_t*)[worker->matchFoundBuf contents];
        *found = 0;

        // Create command buffer and encoder
        id<MTLCommandBuffer> cmdBuf = [worker->queue commandBuffer];
        if (cmdBuf == nil) return 0;

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (encoder == nil) return 0;

        [encoder setComputePipelineState:worker->pipeline];
        [encoder setBuffer:worker->paramsBuf offset:0 atIndex:0];
        [encoder setBuffer:worker->staticWBuf offset:0 atIndex:1];
        [encoder setBuffer:worker->matchFoundBuf offset:0 atIndex:2];
        [encoder setBuffer:worker->matchCounterBuf offset:0 atIndex:3];

        // Calculate grid size
        NSUInteger threadGroupSize = bestThreadGroupSize(worker->maxThreadsPerGroup, worker->threadExecutionWidth);
        if (threadGroupSize > worker->batchSize) threadGroupSize = worker->batchSize;
        if (threadGroupSize == 0) return 0;
        MTLSize gridSize = MTLSizeMake(worker->batchSize, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            NSLog(@"Metal command buffer error: %@", [cmdBuf error]);
            return 0;
        }

        // Read results
        *matchFound = *(int32_t*)[worker->matchFoundBuf contents];
        *matchCounter = *(uint64_t*)[worker->matchCounterBuf contents];

        return (unsigned long)worker->batchSize;
    }
}

void metalFreeWorker(void* handle) {
    @autoreleasepool {
        MetalWorker* worker = (MetalWorker*)handle;
        if (!worker) return;

        CFRelease((__bridge CFTypeRef)worker->matchCounterBuf);
        CFRelease((__bridge CFTypeRef)worker->matchFoundBuf);
        CFRelease((__bridge CFTypeRef)worker->paramsBuf);
        CFRelease((__bridge CFTypeRef)worker->staticWBuf);
        CFRelease((__bridge CFTypeRef)worker->pipeline);
        CFRelease((__bridge CFTypeRef)worker->queue);
        CFRelease((__bridge CFTypeRef)worker->device);
        free(worker);
    }
}

// ==== Tor v3 SHA3-256 + base32 prefix check ====

static NSString* const torV3ShaderSource = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"// Keccak-f[1600] round constants\n"
"constant ulong RC[24] = {\n"
"    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,\n"
"    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,\n"
"    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,\n"
"    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,\n"
"    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,\n"
"    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,\n"
"    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,\n"
"    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL\n"
"};\n"
"\n"
"constant uint RHO[25] = {\n"
"    0, 1, 62, 28, 27, 36, 44, 6, 55, 20,\n"
"    3, 10, 43, 25, 39, 41, 45, 15, 21, 8,\n"
"    18, 2, 61, 56, 14\n"
"};\n"
"\n"
"constant int PI[25] = {\n"
"    0, 10, 20, 5, 15, 16, 1, 11, 21, 6,\n"
"    7, 17, 2, 12, 22, 23, 8, 18, 3, 13,\n"
"    14, 24, 9, 19, 4\n"
"};\n"
"\n"
"constant char B32T[32] = {\n"
"    'a','b','c','d','e','f','g','h','i','j','k','l','m',\n"
"    'n','o','p','q','r','s','t','u','v','w','x','y','z',\n"
"    '2','3','4','5','6','7'\n"
"};\n"
"\n"
"inline ulong rotl64(ulong x, uint n) {\n"
"    return (n == 0) ? x : ((x << n) | (x >> (64 - n)));\n"
"}\n"
"\n"
"struct TorV3Params {\n"
"    uint key_count;\n"
"    uint prefix_len;\n"
"    char prefix[64];\n"
"};\n"
"\n"
"kernel void torv3_check(\n"
"    device const uchar* pubkeys [[buffer(0)]],\n"
"    device const TorV3Params* params [[buffer(1)]],\n"
"    device atomic_int* match_found [[buffer(2)]],\n"
"    device ulong* match_index [[buffer(3)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= params->key_count) return;\n"
"    if (atomic_load_explicit(match_found, memory_order_relaxed) != 0) return;\n"
"\n"
"    device const uchar* pk = pubkeys + gid * 32;\n"
"\n"
"    ulong state[25];\n"
"    for (int i = 0; i < 25; i++) state[i] = 0;\n"
"\n"
"    state[0] = 0x63206e6f696e6f2eUL;\n"
"    state[1] = 0x006d75736b636568UL | ((ulong)pk[0] << 56);\n"
"    state[2] = (ulong)pk[1] | ((ulong)pk[2]<<8) | ((ulong)pk[3]<<16) | ((ulong)pk[4]<<24) |\n"
"              ((ulong)pk[5]<<32) | ((ulong)pk[6]<<40) | ((ulong)pk[7]<<48) | ((ulong)pk[8]<<56);\n"
"    state[3] = (ulong)pk[9] | ((ulong)pk[10]<<8) | ((ulong)pk[11]<<16) | ((ulong)pk[12]<<24) |\n"
"              ((ulong)pk[13]<<32) | ((ulong)pk[14]<<40) | ((ulong)pk[15]<<48) | ((ulong)pk[16]<<56);\n"
"    state[4] = (ulong)pk[17] | ((ulong)pk[18]<<8) | ((ulong)pk[19]<<16) | ((ulong)pk[20]<<24) |\n"
"              ((ulong)pk[21]<<32) | ((ulong)pk[22]<<40) | ((ulong)pk[23]<<48) | ((ulong)pk[24]<<56);\n"
"    state[5] = (ulong)pk[25] | ((ulong)pk[26]<<8) | ((ulong)pk[27]<<16) | ((ulong)pk[28]<<24) |\n"
"              ((ulong)pk[29]<<32) | ((ulong)pk[30]<<40) | ((ulong)pk[31]<<48) | (0x03UL<<56);\n"
"    state[6] = 0x0000000000000006UL;\n"
"    state[16] = 0x8000000000000000UL;\n"
"\n"
"    for (int round = 0; round < 24; round++) {\n"
"        ulong C[5];\n"
"        for (int x = 0; x < 5; x++)\n"
"            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];\n"
"        for (int x = 0; x < 5; x++) {\n"
"            ulong D = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);\n"
"            for (int y = 0; y < 5; y++)\n"
"                state[x + 5*y] ^= D;\n"
"        }\n"
"        ulong B[25];\n"
"        for (int i = 0; i < 25; i++)\n"
"            B[PI[i]] = rotl64(state[i], RHO[i]);\n"
"        for (int y = 0; y < 5; y++) {\n"
"            for (int x = 0; x < 5; x++)\n"
"                state[x + 5*y] = B[x + 5*y] ^ (~B[((x+1)%5) + 5*y] & B[((x+2)%5) + 5*y]);\n"
"        }\n"
"        state[0] ^= RC[round];\n"
"    }\n"
"\n"
"    uchar cksum0 = (uchar)(state[0]);\n"
"    uchar cksum1 = (uchar)(state[0] >> 8);\n"
"\n"
"    uchar payload[35];\n"
"    for (int i = 0; i < 32; i++) payload[i] = pk[i];\n"
"    payload[32] = cksum0;\n"
"    payload[33] = cksum1;\n"
"    payload[34] = 0x03;\n"
"\n"
"    uint prefix_len = params->prefix_len;\n"
"    uint bit_offset = 0;\n"
"    bool match = true;\n"
"    for (uint i = 0; i < prefix_len && match; i++) {\n"
"        uint byte_idx = bit_offset / 8;\n"
"        uint bit_idx = bit_offset % 8;\n"
"        uint val;\n"
"        if (bit_idx <= 3) {\n"
"            val = (payload[byte_idx] >> (3 - bit_idx)) & 0x1f;\n"
"        } else {\n"
"            val = ((payload[byte_idx] << (bit_idx - 3)) | (payload[byte_idx + 1] >> (11 - bit_idx))) & 0x1f;\n"
"        }\n"
"        if (B32T[val] != params->prefix[i]) {\n"
"            match = false;\n"
"        }\n"
"        bit_offset += 5;\n"
"    }\n"
"\n"
"    if (match) {\n"
"        int expected = 0;\n"
"        if (atomic_compare_exchange_weak_explicit(match_found, &expected, 1,\n"
"                memory_order_relaxed, memory_order_relaxed)) {\n"
"            *match_index = (ulong)gid;\n"
"        }\n"
"    }\n"
"}\n";

// ---- Tor v3 Bridge implementation ----

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> pubkeyBuf;
    id<MTLBuffer> paramsBuf;
    id<MTLBuffer> matchFoundBuf;
    id<MTLBuffer> matchIndexBuf;
    NSUInteger batchSize;
    NSUInteger maxThreadsPerGroup;
    NSUInteger threadExecutionWidth;
} MetalTorV3Worker;

typedef struct __attribute__((packed)) {
    uint32_t key_count;
    uint32_t prefix_len;
    char prefix[64];
} TorV3Params;

void* metalNewTorV3Worker(int deviceIndex, const char* prefix, int prefixLen,
                          unsigned long batchSize) {
    @autoreleasepool {
        if (batchSize == 0 || batchSize > UINT32_MAX) return NULL;

        id<MTLDevice> device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && [devices count] > 0 && deviceIndex < (int)[devices count]) {
            device = devices[deviceIndex];
        } else {
            device = MTLCreateSystemDefaultDevice();
        }
        if (device == nil) return NULL;

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:torV3ShaderSource options:options error:&error];
        if (library == nil) {
            NSLog(@"Metal TorV3 shader compile error: %@", error);
            return NULL;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"torv3_check"];
        if (function == nil) {
            NSLog(@"Metal function 'torv3_check' not found");
            return NULL;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            NSLog(@"Metal TorV3 pipeline error: %@", error);
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return NULL;

        TorV3Params params;
        memset(&params, 0, sizeof(params));
        params.key_count = 0;
        params.prefix_len = (uint32_t)prefixLen;
        size_t prefixCopyLen = (size_t)(prefixLen < 64 ? prefixLen : 64);
        memcpy(params.prefix, prefix, prefixCopyLen);

        id<MTLBuffer> pubkeyBuf = [device newBufferWithLength:batchSize * 32
                                                       options:(MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)];
        if (pubkeyBuf == nil) return NULL;

        id<MTLBuffer> paramsBuf = [device newBufferWithBytes:&params
                                                      length:sizeof(TorV3Params)
                                                     options:MTLResourceStorageModeShared];

        int32_t zero = 0;
        id<MTLBuffer> matchFoundBuf = [device newBufferWithBytes:&zero
                                                          length:sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared];

        uint64_t zeroIdx = 0;
        id<MTLBuffer> matchIndexBuf = [device newBufferWithBytes:&zeroIdx
                                                          length:sizeof(uint64_t)
                                                         options:MTLResourceStorageModeShared];

        MetalTorV3Worker* worker = (MetalTorV3Worker*)calloc(1, sizeof(MetalTorV3Worker));
        worker->device = device;
        worker->queue = queue;
        worker->pipeline = pipeline;
        worker->pubkeyBuf = pubkeyBuf;
        worker->paramsBuf = paramsBuf;
        worker->matchFoundBuf = matchFoundBuf;
        worker->matchIndexBuf = matchIndexBuf;
        worker->batchSize = (NSUInteger)batchSize;
        worker->maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
        worker->threadExecutionWidth = [pipeline threadExecutionWidth];

        CFRetain((__bridge CFTypeRef)device);
        CFRetain((__bridge CFTypeRef)queue);
        CFRetain((__bridge CFTypeRef)pipeline);
        CFRetain((__bridge CFTypeRef)pubkeyBuf);
        CFRetain((__bridge CFTypeRef)paramsBuf);
        CFRetain((__bridge CFTypeRef)matchFoundBuf);
        CFRetain((__bridge CFTypeRef)matchIndexBuf);

        return worker;
    }
}

unsigned long metalRunTorV3Batch(void* handle, const unsigned char* pubkeys,
                                  unsigned long keyCount,
                                  int* matchFound, unsigned long* matchIndex) {
    @autoreleasepool {
        MetalTorV3Worker* worker = (MetalTorV3Worker*)handle;
        if (!worker) return 0;
        if (keyCount == 0 || keyCount > (unsigned long)worker->batchSize) return 0;
        if (pubkeys == NULL) return 0;
        if (keyCount > UINT32_MAX) return 0;

        void* pubkeyPtr = [worker->pubkeyBuf contents];
        if (pubkeyPtr == NULL) return 0;
        memcpy(pubkeyPtr, pubkeys, keyCount * 32);

        // Update params
        TorV3Params* params = (TorV3Params*)[worker->paramsBuf contents];
        params->key_count = (uint32_t)keyCount;

        // Reset match_found
        int32_t* found = (int32_t*)[worker->matchFoundBuf contents];
        *found = 0;

        id<MTLCommandBuffer> cmdBuf = [worker->queue commandBuffer];
        if (cmdBuf == nil) return 0;

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (encoder == nil) return 0;

        [encoder setComputePipelineState:worker->pipeline];
        [encoder setBuffer:worker->pubkeyBuf offset:0 atIndex:0];
        [encoder setBuffer:worker->paramsBuf offset:0 atIndex:1];
        [encoder setBuffer:worker->matchFoundBuf offset:0 atIndex:2];
        [encoder setBuffer:worker->matchIndexBuf offset:0 atIndex:3];

        NSUInteger threadGroupSize = bestThreadGroupSize(worker->maxThreadsPerGroup, worker->threadExecutionWidth);
        if (threadGroupSize > keyCount) threadGroupSize = (NSUInteger)keyCount;
        if (threadGroupSize == 0) return 0;
        MTLSize gridSize = MTLSizeMake(keyCount, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            NSLog(@"Metal TorV3 command buffer error: %@", [cmdBuf error]);
            return 0;
        }

        *matchFound = *(int32_t*)[worker->matchFoundBuf contents];
        *matchIndex = *(uint64_t*)[worker->matchIndexBuf contents];

        return (unsigned long)keyCount;
    }
}

void metalFreeTorV3Worker(void* handle) {
    @autoreleasepool {
        MetalTorV3Worker* worker = (MetalTorV3Worker*)handle;
        if (!worker) return;

        CFRelease((__bridge CFTypeRef)worker->pubkeyBuf);
        CFRelease((__bridge CFTypeRef)worker->matchIndexBuf);
        CFRelease((__bridge CFTypeRef)worker->matchFoundBuf);
        CFRelease((__bridge CFTypeRef)worker->paramsBuf);
        CFRelease((__bridge CFTypeRef)worker->pipeline);
        CFRelease((__bridge CFTypeRef)worker->queue);
        CFRelease((__bridge CFTypeRef)worker->device);
        free(worker);
    }
}
