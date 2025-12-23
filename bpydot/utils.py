#!/usr/bin/env python3
############################################################ IDENT(1)
#
# $Title: bpydot utilities $
# $Copyright: 2025 Devin Teske. All rights reserved. $
# pylint: disable=line-too-long
# $FrauBSD$
# pylint: enable=line-too-long
#
############################################################ LICENSE
#
# BSD 2-Clause
#
############################################################ DOCSTRING

"""bpydot utilities"""

############################################################ FUNCTIONS

def plur(n: int, singular: str, plural: str = None) -> str:
    """Return singular or plural form based on count.

    Args:
        n: The count
        singular: Singular form
        plural: Plural form (default: singular + 's')

    Returns:
        Appropriate grammatical form

    Examples:
        >>> plur(1, 'file')
        'file'
        >>> plur(2, 'file')
        'files'
        >>> plur(1, 'batch', 'batches')
        'batch'
        >>> plur(2, 'batch', 'batches')
        'batches'
    """
    if n == 1:
        return singular

    if plural is not None:
        return plural

    # Common special cases
    special_plurals = {
        'batch': 'batches',
        'match': 'matches',
        'entry': 'entries',
        'is': 'are',
        'was': 'were',
    }

    if singular in special_plurals:
        return special_plurals[singular]

    # Default: add 's'
    return f"{singular}s"


################################################################################
# END
################################################################################
