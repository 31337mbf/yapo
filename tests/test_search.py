import pinject
import pytest
from hamcrest import assert_that, calling, raises, empty, has_item, starts_with, matches_regexp, has_length, \
    not_none, is_in

import cifrum as lib
from cifrum._search import _Search
from cifrum.common.enums import SecurityType


def test__handle_the_incorrect_query_string():
    assert_that(calling(lib.search).with_args(query=None), raises(ValueError))
    assert_that(calling(lib.search).with_args(query=42), raises(ValueError))
    assert_that(calling(lib.search).with_args(query=.42), raises(ValueError))
    assert_that(calling(lib.search).with_args(query=[]), raises(ValueError))
    assert_that(lib.search(query=''), empty())

    assert_that(calling(lib.search).with_args(query='microsoft', top=[]), raises(ValueError))
    assert_that(calling(lib.search).with_args(query='microsoft', top=.2), raises(ValueError))
    assert_that(calling(lib.search).with_args(query='microsoft', top='a'), raises(ValueError))
    assert_that(lib.search(query='microsoft', top=-10), empty())


@pytest.mark.parametrize('query, expect_item', [('spy', 'us/SPY'),
                                                ('sber', 'micex/SBER'),
                                                ('0890-94127385', 'mut_ru/0890-94127385')])
def test__search_exact_ticker(query, expect_item):
    def __search(query: str):
        search_results = lib.search(query=query)
        return [fin_sym.identifier_str for fin_sym in search_results]

    assert_that(__search(query=query), has_item(expect_item))


@pytest.mark.parametrize('query, exchange_name', [('nyse', 'NYSE'),
                                                  ('nasdaq', 'NASDAQ'),
                                                  ('micex', 'MICEX')])
def test__search_exchange(query, exchange_name):
    rs = lib.search(query=query)
    for r in rs:
        assert_that(r.exchange, starts_with(exchange_name))


def test__search_custom_query():
    rs = lib.search(query='сбербанк', top=30)
    for r in rs:
        assert_that('{} {}'.format(r.short_name, r.long_name), matches_regexp(r'(?i).*сбербанк.*'))
        assert_that(r.security_type, is_in([SecurityType.STOCK_ETF, SecurityType.MUT]))

    rs = lib.search(query='microsoft', top=30)
    for r in rs:
        assert_that('{} {}'.format(r.short_name, r.long_name), matches_regexp(r'(?i).*microsoft.*'))


@pytest.mark.slow
def test__search_exact_finsym():
    qry = 'micex/SBER'

    obj_graph = pinject.new_object_graph(binding_specs=[lib.BindingSpec()])
    search_instance = obj_graph.provide(_Search)

    r = search_instance._check_finsym_access(query=qry)
    assert_that(r, not_none())
    assert r.identifier_str == 'micex/SBER'

    rs = lib.search(query=qry)
    assert_that(rs, has_length(1))
    assert rs[0].identifier_str == 'micex/SBER'
