"""ForgeLM v2 scenario catalogue -- a genuinely new, sealed evaluation surface.

Why this exists
---------------
v1's test split was used to report the headline result, and has therefore been
seen. Any further development decision made against it would turn it into a
model-selection set. v2 supplies scenario families that no ForgeLM model has
ever trained on, frozen before use.

What is *not* changed
---------------------
The task, the output schema, the prompt, the priority rule and the generation
machinery are all inherited unchanged from v1. Only the scenario catalogue is
new. That is deliberate: a v2 result is comparable to v1 precisely because
nothing except the situations changed.

Not paraphrases
---------------
These are different *situations*, not reworded v1 ones. v1 had
`net_dns_failure`; v2 has `v2_net_ipv6_misconfig` and `v2_net_rogue_ap`. v1 had
`sec_phishing_email`; v2 has `v2_sec_mfa_fatigue` and `v2_sec_insider_download`.
A cross-catalogue similarity scan (`scripts/v2_00_build_dataset.py`) reports the
maximum similarity between any v1 and any v2 ticket, so "not a paraphrase" is
measured rather than asserted.

Priority balance
----------------
v1's test split contained only 3 `medium` examples, because `medium` requires a
rule score of exactly 2. The v1 rule is kept unchanged (so the two datasets
remain comparable); instead the family severities and user-count scales are
chosen so that score-2 combinations occur often:

    severity 1 + 10-49 users      severity 2 + fewer than 10 users
    severity 0 + 50+ users        severity 1 + security + <10 users
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .datagen import (
    TEMPLATE_FAMILIES, ScenarioFamily, USER_SCALE_RANGES, _who_phrase,
    compute_priority, render,
)
from .schema import canonical_json
from .seeding import SEEDS, rng

DATASET_VERSION_V2 = "2.0.0"
CATALOGUE_VERSION = "2.0.0"

_S = ScenarioFamily

# 8 categories x 4 families. Severities and user scales are chosen to spread the
# priority distribution rather than to make the task easy.
FAMILIES_V2: tuple[ScenarioFamily, ...] = (
    # ---------------- access_management ----------------
    _S("v2_am_group_nesting", "access_management", "none", False, 1,
       ("single", "team"),
       ("membership of a nested security group is not granting the rights it should",
        "I am in the correct group but the permissions never take effect",
        "adding people to the parent group does not cascade to the child group"),
       ("signing out and back in makes no difference",
        "the group membership shows correctly in the directory",
        "this worked before the group structure was reorganised"),
       ("could someone check how the groups are nested?",
        "please advise whether the inheritance is configured correctly",
        "let me know if the membership needs applying directly")),

    _S("v2_am_service_account_expiry", "access_management", "none", False, 2,
       ("single", "small"),
       ("a service account password expired overnight and an integration has stopped",
        "the automated data feed is failing to authenticate since the account expired",
        "our scheduled job cannot sign in because the service credential lapsed"),
       ("the account is not set to exempt from the expiry policy",
        "the job ran successfully until the end of last week",
        "the error returned is an authentication failure rather than a timeout"),
       ("can the credential be renewed and exempted?",
        "please advise who owns this service account",
        "we need the feed running again before the next cycle")),

    _S("v2_am_privilege_escalation_request", "access_management", "none", False, 0,
       ("single",),
       ("I need temporary local administrator rights to install a development tool",
        "please grant me elevated rights on my own workstation for a short period",
        "I require admin access to configure a driver for a piece of test equipment"),
       ("my manager has approved this in writing",
        "I only need it for the duration of the project",
        "I understand the rights will be time limited"),
       ("could you advise on the approval process?",
        "please let me know what justification is required",
        "happy to accept a time-limited grant")),

    _S("v2_am_federation_cert_rotation", "access_management", "none", False, 3,
       ("dept",),
       ("the federation signing certificate expires this week and nothing has been rotated",
        "our identity federation with the partner tenant is about to break on certificate expiry",
        "the trust relationship will fail when the current signing certificate lapses"),
       ("the partner has already published their replacement certificate",
        "there is no change request raised to perform the rotation",
        "the expiry date is confirmed in the metadata"),
       ("please schedule the rotation before it lapses",
        "can someone own this before the deadline?",
        "we need a plan for the cutover")),

    # ---------------- hardware ----------------
    _S("v2_hw_fan_noise", "hardware", "laptop", False, 1,
       ("single", "small"),
       ("the cooling fan in my laptop is grinding loudly and running constantly",
        "my machine sounds like it is straining and the fan never stops",
        "there is a rattling noise from the vent whenever the fan spins up"),
       ("the noise started gradually over the past fortnight",
        "the underside gets warm even with nothing demanding running",
        "clearing the vents with compressed air did not help"),
       ("could this be looked at before it fails completely?",
        "please advise whether it needs a service",
        "is a replacement fan something you stock?")),

    _S("v2_hw_scanner_feeder", "hardware", "printer", False, 0,
       ("small", "team"),
       ("the document feeder on the scanner pulls pages through at an angle",
        "scanned pages come out skewed no matter how the originals are loaded",
        "the feeder grabs the corner of each sheet and crops the edge"),
       ("the flatbed glass produces a straight scan every time",
        "it began after the rollers were last cleaned",
        "the guides are set correctly against the paper"),
       ("can someone realign the feeder?",
        "please log a service call if parts are needed",
        "any adjustment we can make ourselves?")),

    _S("v2_hw_ups_battery", "hardware", "none", False, 3,
       ("dept",),
       ("the uninterruptible power supply in the comms room is reporting battery failure",
        "the UPS panel shows a failed battery module and is beeping continuously",
        "our backup power unit has flagged its batteries as end of life"),
       ("a self test was run and the unit did not hold load",
        "the batteries are past their rated replacement date",
        "everything in that rack depends on this unit"),
       ("please arrange replacement batteries",
        "can someone assess the risk to the rack?",
        "we need this addressed before any power event")),

    _S("v2_hw_touchscreen_ghost", "hardware", "laptop", False, 2,
       ("single",),
       ("my touchscreen registers taps I never made and the cursor jumps around",
        "the screen responds to phantom touches and opens things by itself",
        "the display accepts input in the top corner with nothing touching it"),
       ("disabling the touch driver stops the behaviour entirely",
        "the screen is clean and there is no visible damage",
        "it makes the machine effectively unusable for editing"),
       ("can the digitiser be tested?",
        "please advise whether this is a warranty repair",
        "is a loaner available while it is looked at?")),

    # ---------------- network ----------------
    _S("v2_net_ipv6_misconfig", "network", "none", False, 2,
       ("team",),
       ("an IPv6 misconfiguration is causing intermittent name resolution failures",
        "hosts are preferring an IPv6 route that does not work and timing out",
        "connections hang for thirty seconds before falling back to IPv4"),
       ("disabling IPv6 on a test machine resolves the delay completely",
        "it appeared after the router advertisement settings were changed",
        "internal and external destinations are equally affected"),
       ("please review the router advertisements",
        "can the advertised prefix be corrected?",
        "we would like this looked at by the network team")),

    _S("v2_net_bandwidth_shaping", "network", "none", False, 1,
       ("team", "dept"),
       ("the traffic shaping policy is throttling video calls to unusable quality",
        "video meetings freeze and drop while file downloads run at full speed",
        "call quality collapses in the afternoon when the link gets busy"),
       ("a speed test between calls shows plenty of headroom",
        "the shaping rules were revised at the start of the month",
        "audio-only calls are noticeably more stable than video"),
       ("could the shaping classes be reviewed?",
        "please advise whether calls can be prioritised",
        "we would like the previous policy compared")),

    _S("v2_net_rogue_ap", "network", "wifi", True, 2,
       ("team", "dept"),
       ("wireless monitoring has flagged an unauthorised access point broadcasting our network name",
        "an access point we do not own is advertising a network identical to ours",
        "an unknown wireless device is impersonating the corporate network"),
       ("the device does not appear in our asset register",
        "the signal is strongest near the shared reception area",
        "staff devices have been connecting to it automatically"),
       ("please locate and remove the device",
        "can you preserve the monitoring logs?",
        "we need to know whether any credentials were captured")),

    _S("v2_net_cable_damage", "network", "none", False, 3,
       ("dept",),
       ("a contractor has cut through a cable run during building work",
        "the fibre between two buildings was severed during excavation",
        "structural work has damaged the trunk cabling and connectivity is down"),
       ("the affected segment has no redundant path",
        "the contractor has stopped work in that area",
        "the break is confirmed by the cable tester"),
       ("please arrange an emergency splice",
        "we need an estimate for restoration",
        "can someone attend the site?")),

    # ---------------- software ----------------
    _S("v2_sw_macro_blocked", "software", "none", False, 1,
       ("team",),
       ("a policy change has blocked macros and our reporting workbook no longer runs",
        "the spreadsheet automation we depend on is disabled by the new macro policy",
        "macros are now blocked and the month-end workbook cannot calculate"),
       ("the workbook is internally authored rather than downloaded",
        "the block message refers to an untrusted location",
        "moving the file to a local folder did not change the outcome"),
       ("can the location be added as trusted?",
        "please advise on the approved way to sign the workbook",
        "we need a route that satisfies the policy")),

    _S("v2_sw_browser_extension", "software", "none", False, 1,
       ("team",),
       ("a required browser extension was removed by the latest update",
        "the extension our workflow depends on has disappeared after the browser upgrade",
        "the toolbar add-in is no longer present following the update"),
       ("reinstalling it manually is blocked by policy",
        "the extension is still listed as supported by the vendor",
        "colleagues who have not yet updated still have it"),
       ("can the extension be redeployed centrally?",
        "please advise whether it is still approved",
        "we would like it restored through the managed catalogue")),

    _S("v2_sw_api_deprecation", "software", "database", False, 2,
       ("single", "small"),
       ("the vendor has retired the API version our integration uses and it now fails",
        "our data sync stopped when the supplier deprecated the endpoint",
        "calls to the reporting interface return a version-not-supported error"),
       ("the vendor's notice was sent to an address nobody monitors",
        "the replacement endpoint has a different response format",
        "the last successful sync was before the retirement date"),
       ("can someone assess the migration effort?",
        "please advise who owns this integration",
        "we need a plan to move to the supported version")),

    _S("v2_sw_print_driver", "software", "printer", False, 1,
       ("team",),
       ("the print driver is incompatible with the new operating system build",
        "printing fails with a driver error after the operating system upgrade",
        "documents queue and then vanish since the platform update"),
       ("machines still on the previous build print without issue",
        "the manufacturer lists a newer driver as supported",
        "removing and re-adding the printer does not help"),
       ("could the updated driver be packaged?",
        "please advise on a workaround in the meantime",
        "we would like this deployed to the affected group")),

    # ---------------- security ----------------
    _S("v2_sec_mfa_fatigue", "security", "none", True, 2,
       ("single", "small"),
       ("I am receiving repeated multi-factor prompts I did not trigger",
        "someone is spamming my sign-in approvals hoping I will accept one",
        "my phone has been buzzing with authentication requests all evening"),
       ("I have not approved any of them",
        "the requests continued after I put the phone down",
        "the prompts list a location I do not recognise"),
       ("please review the sign-in attempts",
        "should my password be changed immediately?",
        "can the account be protected while this is investigated?")),

    _S("v2_sec_shadow_it", "security", "none", True, 2,
       ("team",),
       ("a team has been storing company documents in an unsanctioned cloud account",
        "work files are being kept in a personal storage service outside our control",
        "an unapproved file-sharing tool is being used for client material"),
       ("the account is registered to a personal address",
        "some of the material appears to be commercially sensitive",
        "nobody raised this through the approvals process"),
       ("please advise on the correct process to follow",
        "can the data be migrated to a sanctioned location?",
        "we need guidance on what to do with the existing files")),

    _S("v2_sec_insider_download", "security", "none", True, 3,
       ("single",),
       ("a departing employee has been downloading unusually large volumes of files",
        "someone serving notice has copied an entire project folder to removable media",
        "an account under notice shows bulk file access far outside its normal pattern"),
       ("their last working day is at the end of this week",
        "the volume is many times their usual daily activity",
        "the access occurred outside normal working hours"),
       ("please preserve the audit logs",
        "can access be restricted pending review?",
        "we need guidance on the correct process here")),

    _S("v2_sec_supply_chain_alert", "security", "none", True, 3,
       ("dept",),
       ("a supplier has notified us of a breach affecting data they hold for us",
        "one of our vendors has disclosed an incident involving our records",
        "a third party processing our data has reported a compromise"),
       ("the notification arrived through the security mailbox this morning",
        "the vendor has not yet confirmed which records are involved",
        "our contract requires notification within a fixed window"),
       ("please assess the scope of exposure",
        "can someone confirm what data they hold?",
        "we need to know our notification obligations")),

    # ---------------- account_billing ----------------
    _S("v2_ab_currency_mismatch", "account_billing", "none", False, 1,
       ("single", "small"),
       ("the invoice has been issued in the wrong currency",
        "we have been billed in dollars against a contract priced in pounds",
        "the currency on this invoice does not match our agreement"),
       ("the exchange rate applied is not the contracted one",
        "previous invoices were issued correctly",
        "finance cannot process it in its current form"),
       ("can this be reissued in the correct currency?",
        "please advise who to contact at the supplier",
        "we need a corrected document before payment")),

    _S("v2_ab_auto_renew_surprise", "account_billing", "none", False, 1,
       ("team",),
       ("a subscription auto-renewed and charged us without any advance notice",
        "we were billed for another year of a tool we intended to cancel",
        "the renewal went through automatically despite our decision to stop"),
       ("no reminder was received before the charge",
        "the cancellation was agreed internally last quarter",
        "the amount has already left the account"),
       ("can we request a refund?",
        "please advise on the cancellation terms",
        "we would like auto-renewal disabled going forward")),

    _S("v2_ab_po_mismatch", "account_billing", "none", False, 0,
       ("single", "small"),
       ("the purchase order number on the invoice does not match the one we raised",
        "our finance system is rejecting the invoice because the order reference is wrong",
        "the supplier has quoted a purchase order we do not recognise"),
       ("the correct reference is recorded in the procurement system",
        "the amounts otherwise agree exactly",
        "this is holding up an otherwise valid payment"),
       ("could the reference be corrected?",
        "please advise how to resubmit",
        "happy to supply the correct order number")),

    _S("v2_ab_tax_code_wrong", "account_billing", "none", False, 0,
       ("team", "dept"),
       ("the wrong tax code has been applied across our recent invoices",
        "sales tax is being charged at a rate that does not apply to us",
        "our exemption is not being reflected on the billing"),
       ("our registration details were provided when the account was set up",
        "finance flagged it during the quarterly review",
        "the difference accumulates across every line"),
       ("can the account be corrected?",
        "please advise whether past invoices can be adjusted",
        "who maintains the tax configuration?")),

    # ---------------- email ----------------
    _S("v2_email_quarantine_release", "email", "email", False, 1,
       ("team",),
       ("legitimate messages from a client are being quarantined by the filter",
        "genuine correspondence keeps landing in quarantine instead of the inbox",
        "the filter is holding back mail we actually need"),
       ("the sender domain is a long-standing customer",
        "releasing individual messages works but is not sustainable",
        "the quarantine reason given is a generic bulk classification"),
       ("could the sender be allow-listed?",
        "please review why these are being held",
        "we would like the rule adjusted")),

    _S("v2_email_shared_mailbox_sync", "email", "email", False, 1,
       ("small",),
       ("a shared mailbox has stopped syncing to the desktop client",
        "the team mailbox shows new items on the web but not in the installed client",
        "our shared inbox is days behind in the desktop application"),
       ("the web interface shows everything correctly",
        "removing and re-adding the mailbox has not resolved it",
        "personal mailboxes on the same machines sync normally"),
       ("can the cached mode settings be checked?",
        "please advise on a rebuild of the local profile",
        "we need the desktop view current again")),

    _S("v2_email_autoreply_loop", "email", "email", False, 2,
       ("team",),
       ("two automatic replies are bouncing between systems and generating hundreds of messages",
        "an out-of-office loop has started between our mailbox and an external system",
        "automatic responses are replying to each other without stopping"),
       ("the volume is climbing steadily",
        "the loop began after an autoresponder was enabled",
        "the messages are filling the shared inbox"),
       ("please break the loop as soon as possible",
        "can the autoresponder be suppressed for that address?",
        "we need the queue cleared")),

    _S("v2_email_attachment_size", "email", "email", False, 0,
       ("single",),
       ("the attachment size limit is blocking a submission I need to send",
        "my document is a few megabytes over the limit and will not send",
        "outgoing mail with the report attached is rejected for size"),
       ("compressing the file has not brought it under the limit",
        "the recipient has asked for it in this format",
        "the deadline for submission is tomorrow"),
       ("is there an approved way to send large files?",
        "could the limit be raised for this message?",
        "please advise on a file transfer option")),

    # ---------------- other ----------------
    _S("v2_oth_visitor_wifi_policy", "other", "none", False, 0,
       ("single",),
       ("I would like clarification on the policy for visitor network access",
        "please confirm what our guests are permitted to do on the visitor network",
        "I need to know the rules before booking an external workshop"),
       ("the intranet page on this looks out of date",
        "the event is a full day with external attendees",
        "I want to brief the organisers correctly"),
       ("could you point me at the current policy?",
        "please advise what to tell attendees",
        "who owns the guest access policy?")),

    _S("v2_oth_relocation_survey", "other", "none", False, 0,
       ("team", "dept"),
       ("we need an IT survey completed ahead of the office relocation",
        "the move to the new floor requires an assessment of the network points",
        "please survey the new space for equipment and connectivity"),
       ("the facilities plan has been finalised",
        "the move is scheduled for the end of next quarter",
        "we need to know what cabling work is required"),
       ("can a survey be scheduled?",
        "please advise on the lead time",
        "who coordinates this with facilities?")),

    _S("v2_oth_printer_consumables", "other", "printer", False, 0,
       ("small", "team"),
       ("we are running low on toner and need replacement consumables ordered",
        "the shared printer is nearly out of supplies and there is no spare",
        "please arrange a stock of cartridges for the floor printer"),
       ("the last order was placed several months ago",
        "the low-supply warning has been showing for a week",
        "we get through roughly one cartridge a month"),
       ("could an order be raised?",
        "please advise on the reordering process",
        "is there a standing order we can join?")),

    _S("v2_oth_power_outage_plan", "other", "none", False, 3,
       ("dept",),
       ("a planned building power outage will affect all on-site equipment",
        "the landlord has scheduled a power shutdown and our systems need protecting",
        "electrical maintenance will cut power to the building for several hours"),
       ("the outage window is confirmed for the coming weekend",
        "the comms room has no generator backup",
        "an uncontrolled shutdown risks data loss"),
       ("we need a controlled shutdown and restart plan",
        "please advise what must be powered down",
        "can someone own the restart sequence?")),
)


# Examples per family. 32 families x 6 = 192 examples.
EXAMPLES_PER_FAMILY_V2 = 6
TARGET_TOTAL_V2 = len(FAMILIES_V2) * EXAMPLES_PER_FAMILY_V2


def generate_dataset_v2() -> list[dict[str, Any]]:
    """Build the v2 dataset deterministically.

    Uses a distinct seed context (`dataset_generation_v2`) so v2 text can never
    coincide with v1 text through a shared RNG stream.
    """
    records: list[dict[str, Any]] = []

    for fam_index, fam in enumerate(FAMILIES_V2):
        for k in range(EXAMPLES_PER_FAMILY_V2):
            r = rng("dataset_generation", "v2", fam.fid, k)

            template_family = TEMPLATE_FAMILIES[
                (fam_index + k) % len(TEMPLATE_FAMILIES)
            ]
            scale = r.choice(fam.user_scales)
            lo, hi = USER_SCALE_RANGES[scale]
            users = r.randint(lo, hi)

            sym = r.choice(fam.symptoms)
            det = r.choice(fam.details)
            ask = r.choice(fam.asks)
            who = _who_phrase(users, r)

            text = render(template_family, sym, det, ask, who, r)
            priority = compute_priority(fam.base_severity,
                                        fam.is_security_incident, users)

            expected = {
                "category": fam.category,
                "priority": priority,
                "affected_service": fam.affected_service,
                "is_security_incident": fam.is_security_incident,
                "users_affected": users,
            }

            records.append({
                "example_id": f"{fam.fid}-{k:02d}",
                "ticket_text": text,
                "expected_output": expected,
                "expected_output_json": canonical_json(expected),
                "category": fam.category,
                "scenario_family": fam.fid,
                "template_family": template_family,
                "user_scale": scale,
                "base_severity": fam.base_severity,
                "symptom_text": sym,
                "detail_text": det,
                "ask_text": ask,
                "scope_text": who,
                "source": "synthetic:forgelm.datagen_v2",
                "generator_version": DATASET_VERSION_V2,
                "catalogue_version": CATALOGUE_VERSION,
                "schema_version": "1.0.0",
                "generation_seed": SEEDS["dataset_generation"],
                "qc_status": "pending",
            })

    return records


def family_table_v2() -> list[dict[str, Any]]:
    return [
        {**asdict(f),
         "symptoms": len(f.symptoms),
         "details": len(f.details),
         "asks": len(f.asks),
         "n_examples": EXAMPLES_PER_FAMILY_V2}
        for f in FAMILIES_V2
    ]
