/* Copyright (C) 2019  Anonymous
 *
 * This is a pre-release version of the DPF++ library distributed anonymously
 * for peer review. A public release of the software will be published under the
 * LPGL v2.1 license in the near future. Please do not redistribute this version
 * of the software.
 */

#ifndef DPFPP_PRG_H__
#define DPFPP_PRG_H__

#include <cstring>

#include "aes.h"

namespace dpf
{

namespace prg
{

struct aes
{
    static void eval(const __m128i & seed, void * outbuf, uint32_t len, uint32_t from = 0)
    {
        static const AES_KEY key;
        __m128i * outbuf128 = reinterpret_cast<__m128i *>(outbuf);
        const auto c = _mm_xor_si128(key.rd_key[0], seed);

        for (uint32_t i = 0; i < len; ++i)
            outbuf128[i] = _mm_xor_si128(c, _mm_set_epi64x(0, from+i));
        for (uint32_t j = 1; j < key.rounds; ++j)
            for (uint32_t i = 0; i < len; ++i)
                outbuf128[i] = _mm_aesenc_si128(outbuf128[i], key.rd_key[j]);
        for (uint32_t i = 0; i < len; ++i)
            outbuf128[i] = _mm_xor_si128(_mm_aesenclast_si128(outbuf128[i], key.rd_key[key.rounds]), seed);
	}
}; // struct aes

/*
template<>
inline void PRG(const LowMC<__m256i> & prgkey, const __m256i seed, void * outbuf, const uint32_t len, const uint32_t from)
{
	__m256i * outbuf256 = reinterpret_cast<__m256i *>(outbuf);
	for (size_t i = 0; i < len; ++i)
	{
		auto tmp = _mm256_xor_si256(seed, _mm256_set_epi64x(0, 0, 0, from+i));
		outbuf256[i] = prgkey.encrypt(tmp);
		outbuf256[i] = _mm256_xor_si256(outbuf256[i], tmp);
	}
} // PRG

template<>
inline void PRG(const LowMC<__m128i> & prgkey, const __m128i seed, void * outbuf, const uint32_t len, const uint32_t from)
{
	__m128i * outbuf128 = reinterpret_cast<__m128i *>(outbuf);
	for (size_t i = 0; i < len; ++i)
	{
		auto tmp = _mm_xor_si128(seed, _mm_set_epi64x(0, from+i));
		outbuf128[i] = prgkey.encrypt(tmp);
		outbuf128[i] = _mm_xor_si128(outbuf128[i], tmp);
	}
} // PRG
*/

} // namespace prg

} // namespace dpf

#endif
