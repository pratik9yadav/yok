Next week we switch on card payments for online orders. The code is in `payments.py` and it works against the gateway's sandbox.
Our logs are shipped to a third-party log service that the whole team and two freelancers can read, and the support dashboard shows `recent_charges()`.
Please do a production review of `payments.py` and fix what you find. The checkout page calls `charge_card(gateway, order_id, amount_paise, card_number, expiry, cvv, phone)` and expects a transaction id back or a `PaymentError`, and support still needs to be able to find a charge by its order id, so keep that working.
The server runs for months without a restart.
