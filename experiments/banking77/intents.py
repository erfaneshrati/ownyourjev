"""Rich intent descriptions used ONLY in the labeling prompt (the trained question stays as in tasks.py).
Written from the intent names, the 20 seeds, and dev-half-A error analysis."""

DESC = {
 "Refund_not_showing_up": "A refund the customer was promised / a merchant issued has not appeared in their account yet.",
 "activate_my_card": "How to activate a (new) card; activation not working.",
 "age_limit": "Minimum age to open/use an account; accounts for children/teenagers.",
 "apple_pay_or_google_pay": "Using Apple Pay / Google Pay / phone wallet with the card, adding the card to it, top-ups via them.",
 "atm_support": "Which ATMs/cash machines the card works at, whether a specific ATM can be used.",
 "automatic_top_up": "Auto top-up feature: what it is, setting it up, turning it off, how it works.",
 "balance_not_updated_after_bank_transfer": "Customer made or received a bank transfer INTO their account (incl. topping up by transfer) and the balance still doesn't show it.",
 "balance_not_updated_after_cheque_or_cash_deposit": "Deposited cash or a cheque but the balance hasn't updated / money not showing.",
 "beneficiary_not_allowed": "Cannot add or pay a beneficiary/payee/recipient; beneficiary rejected or 'not allowed'.",
 "cancel_transfer": "Wants to cancel/stop/reverse a transfer already made (e.g. sent to the wrong account, changed mind).",
 "card_about_to_expire": "Card is expiring / expired; how to get a replacement for an expiring card; cost of renewal.",
 "card_acceptance": "Where the card is accepted / can be used (shops, merchants, online); a merchant not accepting the card in general.",
 "card_arrival": "Customer ordered a card and it HAS NOT ARRIVED yet / is late / wants to track the card already sent ('still waiting on my card', 'track my card').",
 "card_delivery_estimate": "General question about HOW LONG card delivery takes, delivery times, express delivery, choosing delivery date/address BEFORE any lateness ('how long until I get my card', 'how soon can I get a card'). If the card is overdue or they want to track it -> card_arrival.",
 "card_linking": "Linking/connecting/adding a card to the app or account; re-linking/reactivating a card that was found after being reported lost.",
 "card_not_working": "Physical card just doesn't work / stopped working / broken / damaged chip, without being specific to contactless or ATM or one declined payment.",
 "card_payment_fee_charged": "Charged an (unexpected) fee for a card PAYMENT/purchase (not ATM, not transfer, not top-up, not exchange).",
 "card_payment_not_recognised": "A card payment/charge on the statement the customer doesn't recognise or didn't make (purchases). Not cash withdrawals, not direct debits.",
 "card_payment_wrong_exchange_rate": "A card PURCHASE in a foreign currency was converted at a wrong/bad exchange rate.",
 "card_swallowed": "An ATM retained/swallowed/kept/ate the card.",
 "cash_withdrawal_charge": "Fee charged for withdrawing cash at an ATM; ATM fees.",
 "cash_withdrawal_not_recognised": "A cash/ATM withdrawal on the account the customer did not make; someone took cash from their account/card.",
 "change_pin": "How/where to change or reset the PIN (not blocked).",
 "compromised_card": "Card details may be compromised/stolen/hacked/skimmed, someone else may have used it, worried about fraud on the card in general; wants to secure it.",
 "contactless_not_working": "Contactless/tap payments not working.",
 "country_support": "Which countries the service/card/account is available in or supported; using the account while living abroad.",
 "declined_card_payment": "A card payment/purchase was declined/rejected/refused.",
 "declined_cash_withdrawal": "ATM cash withdrawal was declined/refused / could not get cash.",
 "declined_transfer": "A transfer was declined/rejected by the bank.",
 "direct_debit_payment_not_recognised": "Unrecognised direct debit on the account.",
 "disposable_card_limits": "Limits of disposable virtual cards: how many times usable, how many can be created, spending limits on them.",
 "edit_personal_details": "Change personal details: name, address, phone, email, etc.",
 "exchange_charge": "Fees/charges for exchanging currency or for using the card in a foreign currency; is there a charge for exchange.",
 "exchange_rate": "What exchange rate is used / where to see current rates / how rates are determined.",
 "exchange_via_app": "How to exchange currencies / convert money between currencies in the app; holding/exchanging to another currency.",
 "extra_charge_on_statement": "There is an extra/unexpected small charge or fee on the statement the customer does not understand (e.g. 'extra $1 fee').",
 "failed_transfer": "A transfer failed / didn't go through / is not working (not explicitly 'declined').",
 "fiat_currency_support": "Which (fiat) currencies the account supports for holding/exchanging/spending.",
 "get_disposable_virtual_card": "How to get/create a disposable (single-use) virtual card.",
 "get_physical_card": "Getting the physical card and its PIN: where to find the PIN for a new card, receiving a physical card; mostly 'where is my PIN'.",
 "getting_spare_card": "Getting an additional/spare/extra card on the same account, for family members.",
 "getting_virtual_card": "How to get/order a (non-disposable) virtual card; virtual card cost.",
 "lost_or_stolen_card": "Card lost or stolen; what to do; find card.",
 "lost_or_stolen_phone": "Phone lost or stolen (access to the app).",
 "order_physical_card": "How to order a physical (plastic) card, delivery to a country, cost of ordering a card.",
 "passcode_forgotten": "Forgot the app passcode/password / login; reset passcode.",
 "pending_card_payment": "A card payment is still pending.",
 "pending_cash_withdrawal": "A cash withdrawal shows as pending.",
 "pending_top_up": "A top-up is still pending / not yet completed.",
 "pending_transfer": "An outgoing transfer is still pending/processing.",
 "pin_blocked": "PIN blocked / locked after wrong attempts; how to unblock.",
 "receiving_money": "How to receive money / get paid (salary, from others) into the account; receiving payments.",
 "request_refund": "Wants to request/get a refund for a purchase or how refunds work.",
 "reverted_card_payment?": "A card payment was reversed/reverted/returned/cancelled, money came back or a payment disappeared.",
 "supported_cards_and_currencies": "Which cards (Visa/Mastercard/Amex etc.) and currencies can be used to top up / are accepted.",
 "terminate_account": "Close/delete/cancel the account.",
 "top_up_by_bank_transfer_charge": "Is there a fee for topping up by bank transfer.",
 "top_up_by_card_charge": "Fee charged for topping up with a card.",
 "top_up_by_cash_or_cheque": "Can I top up / deposit with cash or a cheque; how.",
 "top_up_failed": "A top-up failed / didn't work / was declined.",
 "top_up_limits": "Limits on top-ups: maximum/minimum amounts, how much/how often can be topped up.",
 "top_up_reverted": "A top-up was reverted/reversed/cancelled/returned after appearing to go through; money taken from card then returned/top-up disappeared.",
 "topping_up_by_card": "How to top up the account using a debit/credit card; card top-up questions.",
 "transaction_charged_twice": "Charged twice / duplicate transaction for the same payment.",
 "transfer_fee_charged": "Charged a fee for a transfer (sending money).",
 "transfer_into_account": "How to transfer/add money into the account from another bank account; top up via transfer.",
 "transfer_not_received_by_recipient": "Sent a transfer but the recipient hasn't received it.",
 "transfer_timing": "How long transfers take to arrive / processing time for transfers.",
 "unable_to_verify_identity": "Customer tried to verify identity but it fails / not working / cannot verify.",
 "verify_my_identity": "How to verify identity, what documents are needed for verification.",
 "verify_source_of_funds": "Questions about source of funds verification / proving where money comes from.",
 "verify_top_up": "Verification of a card top-up: verification code/amount on the card statement for top-up verification.",
 "virtual_card_not_working": "A virtual card is not working / declined.",
 "visa_or_mastercard": "Is the card Visa or Mastercard; choosing card network.",
 "why_verify_identity": "Why identity verification is needed.",
 "wrong_amount_of_cash_received": "ATM gave the wrong amount of cash vs what was debited.",
 "wrong_exchange_rate_for_cash_withdrawal": "Cash withdrawal abroad charged at a wrong exchange rate / ATM foreign withdrawal rate issue.",
}

GUIDANCE = """\
These are BANKING77 intents (a neobank app support dataset). Choose the single most specific intent.
Key distinctions:
- card_arrival = the card is late/not received/tracking it; card_delivery_estimate = how long delivery takes in general.
- top_up_failed = top-up did not work/declined; top_up_reverted = it went through then was reversed/cancelled/returned; pending_top_up = still pending/processing.
- pending_X / declined_X / failed_transfer: pick by the transaction type (card payment, cash withdrawal, top-up, transfer).
- Unrecognised transactions: card purchase -> card_payment_not_recognised; ATM cash -> cash_withdrawal_not_recognised; direct debit -> direct_debit_payment_not_recognised; general fear that the card is compromised/someone has the details -> compromised_card.
- Fees: card purchase -> card_payment_fee_charged; ATM -> cash_withdrawal_charge; transfer -> transfer_fee_charged; top-up by card -> top_up_by_card_charge; top-up by bank transfer -> top_up_by_bank_transfer_charge; currency exchange -> exchange_charge; unexplained extra item on statement -> extra_charge_on_statement.
- Wrong exchange rate: on a purchase -> card_payment_wrong_exchange_rate; on an ATM withdrawal -> wrong_exchange_rate_for_cash_withdrawal; general 'what rate do you use' -> exchange_rate.
- Incoming money missing: transfer -> balance_not_updated_after_bank_transfer; cash/cheque deposit -> balance_not_updated_after_cheque_or_cash_deposit; refund -> Refund_not_showing_up. Outgoing transfer not arrived -> transfer_not_received_by_recipient.
- Holding money in several currencies / which currencies -> fiat_currency_support.
- Ordering "another"/"additional"/"second" card for an existing account -> getting_spare_card; ordering the first physical card -> order_physical_card.
- Whether the service/card is available where the customer lives (e.g. "I live in the US") -> country_support.
- A specific payment refused / card not accepted at a shop just now -> declined_card_payment; general "where can I use my card" -> card_acceptance.
- Replacement for an expiring/expired card, incl. while abroad -> card_about_to_expire.
"""
