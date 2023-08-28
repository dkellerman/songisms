'''GraphQL utilities'''

from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator


def get_paginator(qs, page_size, page, paginated_type, **kwargs):
    p = Paginator(qs, page_size)
    try:
        page_obj = p.page(page)
    except PageNotAnInteger:
        page_obj = p.page(1)
    except EmptyPage:
        page_obj = p.page(p.num_pages)
    return paginated_type(
        page=page_obj.number,
        pages=p.num_pages,
        total=p.count,
        has_next=page_obj.has_next(),
        has_prev=page_obj.has_previous(),
        items=page_obj.object_list,
        **kwargs
    )


def GraphenePaginatedType(id, T):
    import graphene
    return type(id, (graphene.ObjectType,), dict(
        items=graphene.List(T),
        page=graphene.Int(),
        pages=graphene.Int(),
        total=graphene.Int(),
        has_next=graphene.Boolean(),
        has_prev=graphene.Boolean(),
        q=graphene.String(),
    ))
