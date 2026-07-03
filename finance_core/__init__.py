# finance_core — deterministic signal generation and trade ticket construction.
# Intentionally minimal: importing core here causes a RuntimeWarning when the
# package is run with `python -m finance_core.core` because core.py ends up in
# sys.modules twice.  Callers that need produce_ticket should import it directly:
#   from finance_core.core import produce_ticket
from finance_core.ticket import Ticket, build_ticket

__all__ = ["Ticket", "build_ticket"]
