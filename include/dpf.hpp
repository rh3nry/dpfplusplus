#ifndef DPFPP_DPF_HPP__
#define DPFPP_DPF_HPP__

#include <type_traits>  // std::is_same<>
#include <limits>       // std::numeric_limits<>
#include <climits>      // CHAR_BIT
#include <cmath>        // std::log2, std::ceil, std::floor
#include <stdexcept>    // std::runtime_error
#include <array>        // std::array<>
#include <iostream>     // std::istream and std::ostream
#include <vector>       // std::vector<>
#include <memory>       // std::shared_ptr<>
#include <utility>      // std::move
#include <algorithm>    // std::copy
#include <cstring>      // std::memcpy

#include <bsd/stdlib.h> // arc4random_buf
#include <x86intrin.h>  // SSE and AVX intrinsics

#include "dpf++/block.hpp"
#include "dpf++/prg.hpp"

#define L 0
#define R 1

namespace dpf
{

template<typename leaf_t = bool, typename node_t = __m128i, typename prg_type = dpf::prg::aes>
struct dpf_key;

template<typename leaf_t, typename node_t, typename prg_t>
inline leaf_t eval(const dpf_key <leaf_t, node_t, prg_t> & dpfkey, const size_t input);

template<typename leaf_t, typename node_t, typename prg_t>
inline void evalinterval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t from, const size_t to, leaf_t * output, uint8_t * t = NULL);

template<typename leaf_t, typename node_t, typename prg_t>
inline void evalfull(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, leaf_t * output, uint8_t * t = NULL);

template<typename leaf_t, typename node_t, typename prg_t>
inline leaf_t __reeval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t input,
    block<node_t> * S, std::array<block<node_t>, dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf> & final, uint8_t * T, const size_t from_bit);

template<class iterator, typename leaf_t, typename node_t, typename prg_t>
inline std::vector<leaf_t> evallist(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, iterator begin, iterator end, size_t size_hint = 0);

template<typename prg_t, typename node_t>
static inline void expand(const block<node_t> & seed, block<node_t> s[2], uint8_t t[2])
{
    prg_t::eval(clear_lsb(seed, 0b11), s, 2);
    t[L] = get_lsb(s[L]);
    s[L] = clear_lsb(s[L], 0b11);
    t[R] = get_lsb(s[R]);
    s[R] = clear_lsb(s[R], 0b11);
} // dpf::expand

template<typename prg_t, typename node_t>
static inline void traverse2(const block<node_t> & seed,
    const uint8_t cw_t[2], const block<node_t> & cw, const uint8_t prev_t,
    block<node_t> s[2], uint8_t t[2])
{
    prg_t::eval(clear_lsb(seed, 0b11), s, 2);
    t[L] = get_lsb(s[L]) ^ (cw_t[L] & prev_t);;
    s[L] = clear_lsb(xor_if(s[L], cw, !prev_t), 0b11);
    t[R] = get_lsb(s[R]) ^ (cw_t[R] & prev_t);;
    s[R] = clear_lsb(xor_if(s[R], cw, !prev_t), 0b11);
} // dpf::expand

template<typename prg_t, typename node_t>
static inline void traverse(const block<node_t> & seed, const bool direction,
    const uint8_t cw_t, const block<node_t> & cw, const uint8_t prev_t,
    block<node_t> & s, uint8_t & t)
{
    prg_t::eval(clear_lsb(seed, 0b11), &s, 1, direction);
    t = get_lsb(s) ^ (cw_t & prev_t);
    s = clear_lsb(xor_if(s, cw, !prev_t), 0b11);
} // dpf::traverse

template<typename prg_t, typename node_t, size_t nodes_per_leaf>
static inline void stretch_leaf(const block<node_t> & seed, std::array<block<node_t>, nodes_per_leaf> & s)
{
    prg_t::eval(clear_lsb(seed), &s, nodes_per_leaf);
} // dpf::stretch_leaf

template<typename leaf_t, typename node_t, typename prg_type>
struct dpf_key final
{
  public:
    static constexpr size_t bits_per_leaf = std::is_same<leaf_t, bool>::value ? 1 : sizeof(leaf_t) * CHAR_BIT;
    static constexpr bool is_packed = (sizeof(leaf_t) < sizeof(node_t));
    static constexpr size_t leaves_per_node = dpf_key::is_packed ? sizeof(node_t) * CHAR_BIT / bits_per_leaf : 1;
    static constexpr size_t nodes_per_leaf = dpf_key::is_packed ? 1 : std::ceil(static_cast<double>(bits_per_leaf) / (sizeof(node_t) * CHAR_BIT));
    static_assert(leaves_per_node * bits_per_leaf == sizeof(node_t) * CHAR_BIT
        || nodes_per_leaf * sizeof(node_t) == sizeof(leaf_t));

    using block_t = block<node_t>;
    using prg_t = prg_type;
    using finalizer_t = std::array<block_t, nodes_per_leaf>;
    typedef std::pair<finalizer_t, finalizer_t> (*finalizer_callback)(const size_t, const leaf_t &, const block_t[2], const uint8_t[2]);

    inline static constexpr size_t depth(const size_t nitems)
    {
        auto tmp = (nitems > 0) ? static_cast<double>(nitems) : std::pow(2, 64);
        return std::ceil(std::log2(std::ceil(tmp / dpf_key::leaves_per_node)));
    }
    inline constexpr size_t depth() const { return dpf_key::depth(nitems); }

    inline static constexpr size_t input_bits(const size_t nitems) { return (nitems > 0) ? std::ceil(std::log2(nitems)) : 64; }
    inline constexpr size_t input_bits() const { return dpf_key::input_bits(nitems); }

    inline static constexpr size_t nodes_in_interval(const size_t from, const size_t to) { return (to < from) ? 0 : std::max(1.0, std::ceil(static_cast<double>(to+1) / leaves_per_node) - std::floor(static_cast<double>(from) / leaves_per_node)); }

    inline static constexpr size_t interval_bytes(const size_t from, const size_t to) { return nodes_in_interval(from, to) * (is_packed ? sizeof(node_t) : sizeof(leaf_t)); }
    inline constexpr size_t full_bytes() { return interval_bytes(0, nitems-1); }

    inline static constexpr size_t nodes_at_leaf_layer(const size_t nitems)
    {
        auto tmp = (nitems > 0) ? static_cast<double>(nitems) : std::pow(2, 64);
        return std::ceil(tmp / dpf_key::leaves_per_node);
    }
    inline constexpr size_t nodes_at_leaf_layer() const { return dpf_key::nodes_at_leaf_layer(nitems); }

    inline dpf_key(dpf_key &&)                  = default;
    inline dpf_key & operator=(dpf_key &&)      = default;
    inline dpf_key(const dpf_key &)             = default;//delete;
    inline dpf_key & operator=(const dpf_key &) = default;//delete;

    inline bool operator==(const dpf_key & rhs) const { return nitems == rhs.nitems && root == rhs.root && cw == rhs.cw && finalizer == rhs.finalizer; }
    inline bool operator!=(const dpf_key & rhs) const { return !(*this == rhs); }

    // static auto read_dpfkey(std::istream & is)
    // {
    // 	size_t nitems;
    // 	node_t root;
    // 	is >> nitems >> root;

    // 	const size_t depth = dpf_key::depth(nitems);
    // 	std::vector<block_t> cw(depth);
    // 	for (size_t i = 0; i <= depth; ++i) is >> cw[i];

    // 	finalizer_t finalizer;
    // 	is >> finalizer;

    // 	return std::forward<dpf_key>(dpf_key(nitems, std::move(root), std::move(cw), std::move(finalizer)));
    // } // dpf_key::read_dpfkey

    static auto default_make_finalizer(const size_t target, const leaf_t & val, const block_t s[2], [[maybe_unused]] const uint8_t t[2])
    {
        finalizer_t finalizer;

        finalizer_t stretched[2];
        stretch_leaf<prg_t>(s[L], stretched[L]);
        stretch_leaf<prg_t>(s[R], stretched[R]);

        if constexpr(dpf_key::is_packed)
        {
            auto finalizer0 = reinterpret_cast<block<node_t> *>(&finalizer[0]);
            if constexpr(std::numeric_limits<leaf_t>::is_integer)
            {
                if constexpr(std::is_same<leaf_t, bool>::value)
                {
                    *finalizer0 = val ? 1 : 0;
                }
                else
                {
                    typedef typename std::make_unsigned_t<leaf_t> unsigned_leaf_t;
                    *finalizer0 = static_cast<unsigned_leaf_t>(val);
                }
                *finalizer0 = finalizer0->shiftl(bits_per_leaf * (target % leaves_per_node));
            }
            else
            {
                *finalizer0 = val;
            }
        }
        else
        {
            std::memcpy(&finalizer[0], &val, sizeof(finalizer_t));
        }
        for (size_t j = 0; j < nodes_per_leaf; ++j)
        {
            finalizer[j] ^= stretched[L][j] ^ stretched[R][j];
        }
        return std::make_pair(finalizer, finalizer);
    } // dpf_key::default_make_finalizer

    static auto make_shared_finalizer(const size_t target, const leaf_t & val, const block_t s[2], const uint8_t t[2])
    {
        finalizer_t tmp[3];
        stretch_leaf(s[L], tmp[0]);
        stretch_leaf(s[R], tmp[1]);
        arc4random_buf(&tmp[2], sizeof(finalizer_t));

        auto tmp2 = reinterpret_cast<leaf_t *>(tmp);

        return std::make_pair(tmp2[2], t[L] ? (tmp2[1]-tmp2[0])-tmp2[2] : (tmp2[0]-tmp2[1])-tmp2[2]);
    } // dpf_key::make_shared_finalizer

    static auto gen(size_t nitems, size_t target, const block_t & root0, const block_t & root1, const leaf_t & val = 1, const finalizer_callback make_finalizer = default_make_finalizer)
    {
        if (nitems && nitems <= target)
        {
            throw std::runtime_error("target point out of range");
        }
        block_t root[2] = { root0, root1 };
        uint8_t t[2] = { get_lsb(root[0]), !t[0] };
        root[1] = set_lsb(root[1], t[1]);
        block_t s[2] = { root[0], root[1] };

        const size_t depth = dpf_key::depth(nitems);
        std::vector<block_t> cw;
        cw.reserve(depth);

        block_t s0[2], s1[2];
        uint8_t t0[2], t1[2];
        const size_t nbits = input_bits(nitems);
        for (size_t layer = 0; layer < depth; ++layer)
        {
            const uint8_t bit = (target >> (nbits - layer - 1)) & 1U;

            expand<prg_t>(s[0], s0, t0);
            expand<prg_t>(s[1], s1, t1);

            const uint8_t keep = (bit == 0) ? L : R, lose = 1 - keep;
            bool cwt[2] = {
                cwt[L] = t0[L] ^ t1[L] ^ bit ^ 1,
                cwt[R] = t0[R] ^ t1[R] ^ bit
            };
            auto nextcw = s0[lose] ^ s1[lose];

            s[L] = xor_if(s0[keep], nextcw, !t[L]);
            t[L] = t0[keep] ^ (t[L] & cwt[keep]);

            s[R] = xor_if(s1[keep], nextcw, !t[R]);
            t[R] = t1[keep] ^ (t[R] & cwt[keep]);

            cw.emplace_back(set_lsbs(nextcw, cwt));
        }

        auto [finalizer0, finalizer1] = make_finalizer(target, val, s, t);

        return std::make_pair(
            std::forward<dpf_key>(dpf_key(nitems, root[0], cw, finalizer0)),
            std::forward<dpf_key>(dpf_key(nitems, root[1], cw, finalizer1)));

    }
    static auto gen(size_t nitems, size_t target, const leaf_t & val = 1, const finalizer_callback make_finalizer = default_make_finalizer)
    {
        block_t root[2];
        arc4random_buf(root, sizeof(root));
        return gen(nitems, target, root[0], root[1], val, make_finalizer);
    } // dpf_key::gen

    inline leaf_t eval(const size_t input) const { return std::forward<leaf_t>(dpf::eval(*this, input)); }
    inline void evalinterval(const size_t from, const size_t to, leaf_t * output, uint8_t * t = NULL) const { dpf::evalinterval(*this, from, to, output, t); }
    inline void evalfull(leaf_t * output, uint8_t * t = NULL) const { dpf::evalfull(*this, output, t); }
    template<class iterator>
    inline std::vector<leaf_t> evallist(iterator begin, iterator end, size_t size_hint = 0) const { return std::forward<std::vector<leaf_t>>(dpf::evallist(*this, std::forward<iterator>(begin), std::forward<iterator>(end), size_hint)); }

struct interator final
{
    inline explicit interator(const size_t v) : value(v) { }
    inline operator size_t() const { return value; }
    inline size_t operator*() const { return value; }
    inline interator & operator++() { ++value; return *this; }
  private:
    size_t value;
};

template<class Iterator>
struct filterator final
{
  public:
    inline filterator(const dpf_key & dpfkey_, Iterator && it1_, Iterator && it2_)
      : dpfkey(dpfkey_),
        it1(std::move(it1_)),
        it2(std::move(it2_)) { }
    inline auto begin() const { return std::forward<const_iterator<Iterator>>(const_iterator<Iterator>(dpfkey, it1)); }
    inline auto end() const { return std::forward<const_iterator<Iterator>>(const_iterator<Iterator>(dpfkey, it2)); }
  private:
    const dpf_key & dpfkey;
    const Iterator it1;
    const Iterator it2;
};
    template<class Iterator = interator>
    struct const_iterator final
    {
      public:
        typedef dpf_key::const_iterator<Iterator> self_type;
        typedef std::pair<leaf_t, size_t> value_type;
        typedef const value_type & reference;
        typedef void pointer;
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef ssize_t difference_type;

        inline explicit const_iterator(const dpf_key & dpfkey_, Iterator curr_)
          : dpfkey(dpfkey_),
            S(dpfkey.depth()+1),
            T(dpfkey.depth()+1),
            curr(curr_),
            prev(curr_)
        {
            S[0] = dpfkey.root;
            T[0] = get_lsb(dpfkey.root, 0b01);
            val = dpf::__reeval(dpfkey, curr_, S.data(), final, T.data(), 0);
        }
        inline const_iterator(const_iterator &&) = default;
        inline const_iterator & operator=(const_iterator &&) = default;
        inline const_iterator(const const_iterator &) = default;
        inline const_iterator & operator=(const const_iterator &) = default;
        inline ~const_iterator() = default;

        inline self_type & operator++() { ++curr; return *this; }
        inline self_type operator++(const int) { auto copy(*this); ++curr; return std::move(copy); }
        inline self_type & operator--() { --curr; return *this; }
        inline self_type operator--(const int) { auto copy(*this); --curr; return std::move(copy); }

        inline bool operator==(const self_type & rhs) const { return dpfkey == rhs.dpfkey && curr == rhs.curr; }
        inline bool operator!=(const self_type & rhs) const { return !(*this == rhs); }

        inline auto operator*()
        {
            auto curr_ = *curr;
            if (curr_ != prev)
            {
                auto from_bit = __builtin_clzll(curr_ ^ prev) - 64 + dpfkey.input_bits();
                val = dpf::__reeval(dpfkey, curr_, S.data(), final, T.data(), from_bit);
                prev = curr_;
            }
            return std::forward<value_type>(std::make_pair(val, curr_));
        }
      private:
        const dpf_key & dpfkey;
        std::vector<block_t> S;
        std::vector<uint8_t> T;
        finalizer_t final;
        Iterator curr;
        size_t prev;
        leaf_t val;
    };

    inline auto begin() const { return std::forward<const_iterator<interator>>(const_iterator<interator>(*this, interator(0))); }
    inline auto cbegin() const { return std::forward<const_iterator<interator>>(const_iterator<interator>(*this, interator(0))); }
    inline auto end() const { return std::forward<const_iterator<interator>>(const_iterator<interator>(*this, interator(nitems))); }
    inline auto cend() const { return std::forward<const_iterator<interator>>(const_iterator<interator>(*this, interator(nitems))); }

    inline auto rbegin() const { return std::reverse_iterator<const_iterator<interator>>(end()); }
    inline auto crbegin() const { return std::reverse_iterator<const_iterator<interator>>(end()); }
    inline auto rend() const { return std::reverse_iterator<const_iterator<interator>>(begin()); }
    inline auto crend() const { return std::reverse_iterator<const_iterator<interator>>(begin()); }

    inline auto filtered_by(size_t begin, size_t end) { return filterator<interator>(*this, interator(begin), interator(end)); }
    template<class Iterator>
    inline auto filtered_by(Iterator && begin, Iterator && end) { return filterator<Iterator>(*this, std::move(begin), std::move(end)); }
    template<class Container>
    inline auto filtered_by(const Container & c) { return filterator<typename Container::iterator_type>(*this, std::cbegin(c), std::cend(c)); }

    inline auto refinalize(const finalizer_t & new_finalizer) const
    {
        return std::forward<dpf_key>(nitems, root, cw, new_finalizer);
    }

    const size_t nitems;
    const block_t root;
    const std::vector<block_t> cw;
    const finalizer_t finalizer;

    // template <typename prg_t>
    auto static gen_sub_dpfs(const dpf_key & root_key) {
        const std::vector<block_t> new_cw(++(root_key.cw.begin()), root_key.cw.end());
        const size_t new_nitems = root_key.nitems / 2;

        block_t s[2];
        uint8_t t[2];
        uint8_t cw_t[2] = {
                get_lsb(root_key.cw[0], 0b01),
                get_lsb(root_key.cw[0], 0b10)
            };
        uint8_t advice  = get_lsb(root_key.root, 0b01);
        // dpf_key::traverse2<prg_type, node_t>(
        dpf::traverse2<prg_t>(
                root_key.root,
                cw_t, // NOTE: What is this argument for?
                root_key.cw[0],
                advice,
                s, t // Do we need t or is s all we need?
            );
        std::array<block_t, 2> new_roots = {
                s[0] ^ block_t(t[0]),
                s[1] ^ block_t(t[1])
            };

        return std::make_pair(
            dpf_key(new_nitems, new_roots[0], new_cw, root_key.finalizer),
            dpf_key(new_nitems, new_roots[1], new_cw, root_key.finalizer)
        );
    }

  private:
    dpf_key(size_t nitems_, const block_t & root_, const std::vector<block_t> cw_,
        const finalizer_t & finalizer_)
      : nitems(nitems_),
        root(root_),
        cw(cw_),
        finalizer(finalizer_) { }
}; // struct dpf::dpf_key

// template<typename T>
// inline std::ostream & operator<<(std::ostream & os, const std::vector<T> & vec)
// {
// 	for (T item : vec) os << item;
// 	return os;
// }

// inline std::ostream & operator<<(std::ostream & os, const AES_KEY & prg)
// {
// 	return os.write(reinterpret_cast<const char *>(&prg.rd_key[0]), sizeof(__m128i));
// }
/*
template<typename __mX>
inline std::ostream & operator<<(std::ostream & os, const LowMC<__mX> & prg)
{
    return os.write(reinterpret_cast<const char *>(&prg.key), sizeof(__m128i));
}
*/
template<typename leaf_t, typename node_t, typename prg_t>
inline std::ostream & operator<<(std::ostream & os, const dpf_key<leaf_t, node_t, prg_t> & dpfkey)
{
    return os << dpfkey.nitems << dpfkey.root << dpfkey.cw << dpfkey.finalizer;
}

template<typename leaf_t, typename node_t>
inline leaf_t getword(const block<node_t> & S, const size_t input)
{
    auto S_ = reinterpret_cast<const leaf_t *>(&S);
    if constexpr(sizeof(leaf_t) >= sizeof(node_t)) return *S_;

    return S_[input];
} // dpf::getword

template<>
inline bool getword(const block<__m128i> & S, const size_t input)
{
    const __m128i mask = bool128_mask[input / 64];
    __m128i vcmp = _mm_xor_si128(_mm_and_si128(S >> (input % 64), mask), mask);

    return static_cast<bool>(_mm_testz_si128(vcmp, vcmp));
} // dpf::getword<__m128i,bool>

template<>
inline bool getword(const block<__m256i> & S, const size_t input)
{
    const __m256i mask = bool256_mask[input / 64];
    __m256i vcmp = _mm256_xor_si256(_mm256_and_si256(S >> (input % 64), mask), mask);

    return static_cast<bool>(_mm256_testz_si256(vcmp, vcmp));
} // dpf::getword<__m256i,bool>

template<typename prg_t, typename leaf_t, typename node_t>
inline void finalize(const std::array<block<node_t>, dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf> & finalizer, leaf_t * output, const block<node_t> * s, size_t nnodes, uint8_t * t)
{
    auto output_ = reinterpret_cast<std::array<block<node_t>, dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf> *>(output);
    for (size_t i = 0; i < nnodes; ++i)
    {
        stretch_leaf<prg_t>(s[i], output_[i]);
        for (size_t j = 0; j < dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf; ++j)
        {
            output_[i][j] = xor_if(output_[i][j], finalizer[j], t[i]);
        }
    }
} // dpf::finalize

template<typename leaf_t, typename node_t, typename prg_t>
inline void __evalinterval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t from, const size_t to, leaf_t * output, uint8_t * _t)
{
    const auto nodes_per_leaf = dpfkey.nodes_per_leaf;
    const auto depth = dpfkey.depth();
    const auto nbits = dpfkey.input_bits();
    const auto nodes_in_interval = dpfkey.nodes_in_interval(from, to);

    const size_t from_node = std::floor(static_cast<double>(from) / nodes_per_leaf);

    block<node_t> * s[2] = {
        reinterpret_cast<block<node_t> *>(output) + nodes_in_interval * (nodes_per_leaf - 1),
        s[0] + nodes_in_interval / 2
    };
    uint8_t * t[2] = { _t, _t + nodes_in_interval / 2};

    int curlayer = depth % 2;

    s[curlayer][0] = dpfkey.root;
    t[curlayer][0] = get_lsb(dpfkey.root, 0b01);

    for (size_t layer = 0; layer < depth; ++layer)
    {
        auto & cw = dpfkey.cw[layer];
        uint8_t cw_t[2] = { get_lsb(cw, 0b01), get_lsb(cw, 0b10) };
        curlayer = 1-curlayer;

        size_t i=0, j=0;
        auto nextbit = (from_node >> (nbits-layer-1)) & 1;
        size_t nodes_in_prev_layer = std::ceil(static_cast<double>(nodes_in_interval) / (1ULL << (depth-layer)));
        size_t nodes_in_cur_layer = std::ceil(static_cast<double>(nodes_in_interval) / (1ULL << (depth-layer-1)));

        if (nextbit == 1) traverse<prg_t>(s[1-curlayer][0], R, cw_t[R], cw, t[1-curlayer][j], s[curlayer][0], t[curlayer][0]);
        for (i = nextbit, j = nextbit; j < nodes_in_prev_layer-1; ++j, i+=2)
        {
            traverse2<prg_t>(s[1-curlayer][j], cw_t, cw, t[1-curlayer][j], &s[curlayer][i], &t[curlayer][i]);
        }
        if (nodes_in_prev_layer > j)
        {
            if (i < nodes_in_cur_layer - 1) traverse2<prg_t>(s[1-curlayer][j], cw_t, cw, t[1-curlayer][j], &s[curlayer][i], &t[curlayer][i]);
            else traverse<prg_t>(s[1-curlayer][j], L, cw_t[L], cw, t[1-curlayer][j], s[curlayer][i], t[curlayer][i]);
        }
    }
    finalize<prg_t>(dpfkey.finalizer, output, s[0], nodes_in_interval, t[0]);
} // dpf::__evalinterval


template<typename leaf_t, typename node_t, typename prg_t>
inline void evalinterval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t from, const size_t to, leaf_t * output, uint8_t * t)
{
    uint8_t * tt = t ? t : reinterpret_cast<uint8_t *>(malloc(dpfkey.nodes_in_interval(from, to) * sizeof(uint8_t)));
    __evalinterval(dpfkey, from, to, output, tt);
    if (!t) free(tt);
} // dpf::evalinterval

template<typename leaf_t, typename node_t, typename prg_t>
inline void evalfull(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, leaf_t * output, uint8_t * t)
{
    uint8_t * tt = t ? t : reinterpret_cast<uint8_t *>(malloc(dpfkey.nodes_at_leaf_layer() * sizeof(uint8_t)));
    __evalinterval(dpfkey, 0, dpfkey.nitems-1, output, tt);
    if (!t) free(tt);
} // dpf::evalfull

template<typename leaf_t, typename node_t, typename prg_t>
inline leaf_t eval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t input)
{
    const auto depth = dpfkey.depth();
    const auto nbits = dpfkey.input_bits();

    block<node_t> S = dpfkey.root;
    uint8_t T = get_lsb(dpfkey.root, 0b01);

    for (size_t layer = 0; layer < depth; ++layer)
    {
        auto & cw = dpfkey.cw[layer];
        const uint8_t nextbit = (input >> (nbits-layer-1)) & 1;
        traverse<prg_t>(S, nextbit, get_lsb(cw, nextbit ? 0b10 : 0b01), cw, T, S, T);
    }
    std::array<node_t, dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf> final;
    finalize<prg_t>(dpfkey.finalizer, &final, &S, 1, &T);

    if constexpr(dpfkey.is_packed)
    {
        auto S_ = reinterpret_cast<block<node_t> *>(&final);
        return std::forward<leaf_t>(getword<leaf_t>(*S_, input % dpfkey.leaves_per_node));
    }
    else
    {
        auto ret = reinterpret_cast<leaf_t *>(&final);
        return *ret;
    }
} // dpf::eval

template<typename leaf_t, typename node_t, typename prg_t>
inline leaf_t __reeval(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, const size_t input,
    block<node_t> * S, std::array<block<node_t>, dpf_key<leaf_t, node_t, prg_t>::nodes_per_leaf> & final, uint8_t * T, const size_t from_bit)
{
    const auto depth = dpfkey.depth();
    const auto nbits = dpfkey.input_bits();

    for (auto layer = from_bit; layer < depth; ++layer)
    {
        auto & cw = dpfkey.cw[layer];
        const uint8_t nextbit = (input >> (nbits-layer-1)) & 1;
        traverse<prg_t>(S[layer], nextbit, get_lsb(cw, nextbit ? 0b10 : 0b01), cw, T[layer], S[layer+1], T[layer+1]);
    }
    if (from_bit != depth) finalize<prg_t>(dpfkey.finalizer, &final, &S[depth], 1, &T[depth]);

    if constexpr(dpfkey.is_packed)
    {
        auto S_ = reinterpret_cast<block<node_t> *>(&final);
        return std::forward<leaf_t>(getword<leaf_t>(*S_, input % dpfkey.leaves_per_node));
    }
    else
    {
        auto ret = reinterpret_cast<leaf_t *>(&final);
        return *ret;
    }
} // dpf::__reeval

template<class iterator, typename leaf_t, typename node_t, typename prg_t>
inline std::vector<leaf_t> evallist(const dpf_key<leaf_t, node_t, prg_t> & dpfkey, iterator begin, const iterator end, size_t size_hint)
{
    const auto depth = dpfkey.depth();
    const auto nbits = dpfkey.input_bits();

    std::vector<leaf_t> result;
    result.reserve(size_hint ? size_hint : std::distance(begin, end));
    block<node_t> * S = (block<node_t> *)std::aligned_alloc(sizeof(node_t), sizeof(node_t) * (depth+1));
    uint8_t * T = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * (depth+1)));

    S[0] = dpfkey.root;
    T[0] = get_lsb(dpfkey.root, 0b01);
    std::array<block<node_t>, dpfkey.nodes_per_leaf> final = { 0 };

    auto it = begin;
    result.emplace_back(std::forward<leaf_t>(__reeval(dpfkey, *it, S, final, T, 0)));
    auto prev = *it;
    while (++it != end)
    {
        size_t from_bit = __builtin_clzll(*it ^ prev) - 64 + nbits;
        result.emplace_back(std::forward<leaf_t>(__reeval(dpfkey, *it, S, final, T, from_bit)));
        prev = *it;
    }

    free(S);
    free(T);
    result.shrink_to_fit();
    return std::move(result);
} // dpf::evallist

} // namespace dpf

#endif // DPFPP_DPF_HPP__
