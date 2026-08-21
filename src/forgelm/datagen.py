"""Synthetic dataset generation for ForgeLM.

Why synthetic? Real helpdesk tickets are personal data. Generating our own gives
us (a) a clean licence story, (b) ground-truth labels that are correct *by
construction* rather than by annotator agreement, and (c) explicit control over
the generalisation axis we want to test.

Structure of the generator
--------------------------
A **scenario family** is a recurring real-world situation ("the VPN client keeps
dropping"). It fixes the semantic label fields. A **template family** is a
surface style ("terse", "frustrated", "formal ticket") and carries no label
information. Every example is one (scenario family x template family) pairing
with sampled symptom/detail/ask clauses and a sampled number of affected users.

The split is group-aware on *scenario family*, so the test set contains
situations the model has never been trained on. Template families deliberately
span splits: they are surface styles, and holding them out would measure
"can it handle a new writing style", which is not the research question.
That choice is documented in DATASET_CARD.md and checked by an explicit
cross-split near-duplicate scan.

Label determination
-------------------
`category`, `affected_service` and `is_security_incident` come from the scenario
family. `users_affected` is sampled and then *written into the text* so it is
always recoverable by a reader. `priority` is computed from a fixed, documented
rule, so it is learnable from the text rather than memorised per family.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .seeding import rng, SEEDS
from .schema import canonical_json

DATASET_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Priority rule -- documented, deterministic, inferable from the ticket text
# --------------------------------------------------------------------------

USER_BUMP_LARGE = 50
USER_BUMP_MEDIUM = 10


def compute_priority(base_severity: int, is_security_incident: bool,
                     users_affected: int) -> str:
    """Map (severity, security flag, blast radius) -> priority.

    Deliberately simple and *stated in the dataset card* so a reader can verify
    every label by hand. The model is never shown this rule; it has to infer it
    from ~170 training examples, which is exactly the adaptation we measure.
    """
    score = base_severity
    if is_security_incident:
        score += 1
    if users_affected >= USER_BUMP_LARGE:
        score += 2
    elif users_affected >= USER_BUMP_MEDIUM:
        score += 1

    if score >= 5:
        return "critical"
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


# --------------------------------------------------------------------------
# Scenario families
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioFamily:
    fid: str
    category: str
    affected_service: str
    is_security_incident: bool
    base_severity: int          # 0..3
    user_scales: tuple[str, ...]
    symptoms: tuple[str, ...]
    details: tuple[str, ...]
    asks: tuple[str, ...]


# Population sizes per scale. Sampled uniformly from the inclusive range.
USER_SCALE_RANGES: dict[str, tuple[int, int]] = {
    "single": (1, 1),
    "small": (2, 8),
    "team": (11, 35),
    "dept": (55, 190),
}

_S = ScenarioFamily

FAMILIES: tuple[ScenarioFamily, ...] = (
    # ---------------- access_management ----------------
    _S("am_password_reset", "access_management", "none", False, 0,
       ("single", "team"),
       ("my Windows password expired overnight and the reset page just spins",
        "I can no longer sign in after the password policy change",
        "the self-service password reset tool rejects every new password I try",
        "my domain password stopped working this morning with no warning"),
       ("I already cleared the browser cache and tried a different browser",
        "this started right after the maintenance window last night",
        "the error just says 'unable to process request'",
        "I have a client presentation at 2pm and cannot get into my machine"),
       ("could someone reset it for me?",
        "please advise on next steps",
        "any chance of a quick fix?",
        "let me know what you need from my side")),

    _S("am_mfa_device_lost", "access_management", "none", False, 1,
       ("single",),
       ("I lost the phone that had my authenticator app on it",
        "my replacement handset does not have the MFA tokens migrated",
        "the authenticator app was wiped when my phone was factory reset",
        "I cannot approve the sign-in prompt because the device is gone"),
       ("I still have my password, it is only the second factor that is blocking me",
        "the backup codes were saved on the same device",
        "I reported the handset as lost to facilities yesterday",
        "I am working from a loaner laptop in the meantime"),
       ("can you re-enrol my MFA?",
        "please help me get back in",
        "what is the process to reset this?",
        "who do I need to verify with?")),

    _S("am_account_lockout", "access_management", "none", False, 1,
       ("single", "small"),
       ("my account keeps locking itself every few minutes",
        "I am locked out again despite not typing my password wrong",
        "the account lock keeps re-triggering after each unlock",
        "something on the network appears to be retrying an old password"),
       ("I suspect an old session on my tablet is still using the previous password",
        "it unlocks fine, then locks again within about five minutes",
        "this began after I changed my password on Friday",
        "the event log entry mentions a stale credential"),
       ("can you find what is causing the lockouts?",
        "please clear the lock and check the source",
        "any way to trace which device is doing this?",
        "help would be appreciated")),

    _S("am_permission_request", "access_management", "crm", False, 0,
       ("single", "small"),
       ("I need read access to the pipeline reports module in the CRM",
        "my new role requires edit rights on the accounts section of the CRM",
        "I cannot open the shared opportunity dashboard in the CRM",
        "please grant me the sales-reporting role in the CRM"),
       ("my manager Priya has approved this in writing",
        "I moved from support into the sales operations team last week",
        "the request has been sitting in the approvals queue for four days",
        "I only need view rights, not the ability to edit records"),
       ("could you action the request?",
        "please let me know once it is granted",
        "what approval do you still need?",
        "happy to provide the ticket reference")),

    _S("am_sso_failure", "access_management", "none", False, 2,
       ("team", "dept"),
       ("single sign-on is bouncing users into a redirect loop",
        "the SSO portal returns an error page instead of signing people in",
        "nobody can authenticate through the identity provider right now",
        "SSO logins fail with 'invalid assertion' for everyone who tries"),
       ("direct logins to individual apps still work, it is only SSO that fails",
        "it started at roughly 08:40 this morning",
        "clearing cookies makes no difference",
        "the status page shows everything as green, which does not match what we see"),
       ("please look at the identity provider urgently",
        "can someone check the federation config?",
        "we need an ETA to share with the floor",
        "escalating because it is blocking work")),

    _S("am_offboarding_access", "access_management", "none", False, 1,
       ("single", "small"),
       ("please revoke all system access for a contractor whose engagement ended",
        "we need accounts disabled for staff who left at the end of last month",
        "a departed employee still appears in the active directory group",
        "offboarding was not completed for someone who finished on Friday"),
       ("HR has confirmed the leaving date and the final day was Friday",
        "their line manager has already collected the laptop and badge",
        "the checklist was submitted but nothing seems to have happened",
        "we want this closed before the quarterly access review"),
       ("can you confirm once the accounts are disabled?",
        "please process the offboarding",
        "let me know if you need the HR reference",
        "we need this tidied up for audit")),

    _S("am_ex_employee_active", "access_management", "none", True, 3,
       ("single", "small"),
       ("an account belonging to someone who left in March was used to sign in last night",
        "a terminated employee's credentials are still active and showing recent logins",
        "we found sign-in activity on a disabled-user account that should not exist",
        "someone authenticated using a leaver's account from an unfamiliar location"),
       ("the login came from an IP range we do not recognise",
        "HR confirmed the person left the business several months ago",
        "the session lasted around twenty minutes and touched a file share",
        "no ticket exists explaining why the account was re-enabled"),
       ("please disable it immediately and preserve the logs",
        "we need this investigated today",
        "can you pull the full sign-in history?",
        "treat this as urgent")),

    # ---------------- hardware ----------------
    _S("hw_laptop_wont_boot", "hardware", "laptop", False, 2,
       ("single", "small"),
       ("my laptop powers on but never gets past the manufacturer logo",
        "the machine shows a blue screen and reboots in a loop",
        "the laptop will not start at all, no lights and no fan",
        "it hangs on 'preparing automatic repair' every single time"),
       ("I have tried holding the power button for thirty seconds with no change",
        "it was working normally when I shut it down last night",
        "there was no drop or spill that I am aware of",
        "the external drive with my work is fine, it is only the laptop"),
       ("can I get a loaner while this is looked at?",
        "please advise where to bring it",
        "is a swap possible today?",
        "what are my options?")),

    _S("hw_battery_swelling", "hardware", "laptop", False, 3,
       ("single",),
       ("the battery in my laptop has visibly swollen and lifted the trackpad",
        "my laptop case is bulging and the machine is very hot to the touch",
        "the underside of the laptop has warped and it no longer sits flat",
        "the battery compartment is expanding and the chassis is separating"),
       ("I have unplugged it and moved it away from anything flammable",
        "there is a faint chemical smell coming from the vents",
        "I stopped using it as soon as I noticed",
        "it is currently sitting on a metal filing cabinet"),
       ("what should I do with it right now?",
        "please arrange safe disposal and a replacement",
        "can someone collect it today?",
        "advise urgently on handling")),

    _S("hw_printer_jam", "hardware", "printer", False, 0,
       ("small", "team"),
       ("the printer on the second floor jams on every job",
        "paper keeps getting stuck in the finisher unit of the shared printer",
        "the multifunction printer reports a jam even when the tray is empty",
        "the printer grabs several sheets at once and then stops"),
       ("we have already cleared the visible paper from trays one and two",
        "it started after the paper was refilled yesterday",
        "the display shows an error code we cannot clear",
        "the smaller printer by the kitchen is still working"),
       ("can someone come and look at it?",
        "please log a call with the supplier",
        "is there a service contract we can use?",
        "any quick fix we can try?")),

    _S("hw_docking_station", "hardware", "laptop", False, 1,
       ("single", "small"),
       ("my docking station no longer passes video to either monitor",
        "the dock charges the laptop but the USB ports are dead",
        "the dock disconnects and reconnects every few seconds",
        "nothing happens when I plug into the dock at my desk"),
       ("plugging the monitor directly into the laptop works fine",
        "I tried the dock at a colleague's desk and it behaved the same way",
        "the firmware updater says the dock is already up to date",
        "it worked until the office move last week"),
       ("could I get a replacement dock?",
        "please advise whether to swap it",
        "any known issue with this model?",
        "happy to bring it to the service desk")),

    _S("hw_monitor_dead", "hardware", "none", False, 1,
       ("single", "small"),
       ("my second monitor has no picture at all, just a power light",
        "the display flickers and then goes black after a few minutes",
        "one of the desk monitors shows vertical coloured lines across the panel",
        "the monitor reports 'no signal' regardless of which cable I use"),
       ("I have swapped the cable and tested a different port",
        "the same monitor fails on a colleague's laptop as well",
        "it was fine when I left on Friday",
        "the other screen at the same desk is working normally"),
       ("can it be replaced from stock?",
        "please let me know the process for a swap",
        "is there a spare I can collect?",
        "who should I return the faulty one to?")),

    _S("hw_headset_mic", "hardware", "phone", False, 0,
       ("single", "small"),
       ("my headset microphone is not picked up on calls",
        "callers say I sound muffled and distant on the headset",
        "the headset mute button has stopped responding",
        "only one ear cup produces sound on my headset"),
       ("the built-in laptop microphone works fine, so it is the headset",
        "I have already reinstalled the audio driver",
        "it happens on both the softphone and video calls",
        "the headset is about two years old"),
       ("could I get a replacement?",
        "please advise if these are still in stock",
        "is there a preferred model I should request?",
        "happy to collect one from the service desk")),

    _S("hw_stolen_laptop", "hardware", "laptop", True, 3,
       ("single",),
       ("my work laptop was stolen from my car last night",
        "someone took my laptop bag from the train and the machine was inside",
        "my company laptop has been taken from a cafe table while I stepped away",
        "my laptop was removed from a hotel room during my stay"),
       ("it was signed in to my account and the email client was open",
        "I have reported it to the police and have a crime reference number",
        "the drive was encrypted as far as I know",
        "I had cached files from the shared drive on the desktop"),
       ("please wipe it remotely as soon as possible",
        "can you lock the device and my sessions right away?",
        "what do I need to do about my credentials?",
        "treat this as urgent please")),

    # ---------------- network ----------------
    _S("net_vpn_drops", "network", "vpn", False, 2,
       ("single", "small", "team"),
       ("the VPN client disconnects every few minutes and has to be reconnected",
        "VPN sessions drop whenever a large file transfer starts",
        "the VPN connects, holds for about a minute, then silently drops",
        "remote workers keep losing the VPN tunnel mid-call"),
       ("home broadband looks stable, other sites load without issue",
        "it started after the client was updated to the latest version",
        "switching from wifi to a wired connection makes no difference",
        "the logs show a rekey failure just before each disconnect"),
       ("please investigate the concentrator",
        "can someone check the VPN gateway logs?",
        "is a rollback of the client possible?",
        "we need this stable before month end")),

    _S("net_wifi_dead_zone", "network", "wifi", False, 1,
       ("small", "team"),
       ("there is no wifi signal at all in the north-east corner of the floor",
        "wifi drops out completely in the meeting rooms along the back wall",
        "devices show full bars but no data passes in the breakout area",
        "the wifi will not associate anywhere near the new partition wall"),
       ("wired connections at the same desks are perfectly fine",
        "it appeared after the desks were reconfigured last month",
        "both phones and laptops are affected in that area",
        "moving ten metres towards the lifts restores the signal"),
       ("could someone survey the coverage?",
        "please look at adding an access point",
        "any chance of a site survey this week?",
        "we would like an update for the floor manager")),

    _S("net_slow_internet", "network", "none", False, 1,
       ("team", "dept"),
       ("internet speeds have dropped to a crawl since this morning",
        "web pages take thirty seconds or more to load across the office",
        "downloads are running at a fraction of the usual speed",
        "everything external is slow while internal systems feel normal"),
       ("a speed test shows a fraction of our contracted bandwidth",
        "it is worst between about ten and midday",
        "restarting individual machines makes no difference",
        "the ISP portal does not show any reported incident"),
       ("can someone check the circuit utilisation?",
        "please raise it with the ISP if needed",
        "is anything saturating the link?",
        "we need an update to share with the team")),

    _S("net_dns_failure", "network", "none", False, 3,
       ("dept",),
       ("internal name resolution has failed and nothing resolves by hostname",
        "DNS lookups are timing out across the whole site",
        "systems can reach IP addresses directly but no hostnames resolve",
        "every application that depends on name resolution is failing"),
       ("connecting by raw IP address still works, which points at DNS",
        "it began immediately after the overnight change window",
        "both primary and secondary resolvers appear unreachable",
        "line-of-business applications are all reporting connection errors"),
       ("this is stopping work across the site",
        "please engage the network team immediately",
        "we need someone on this now",
        "can we get a bridge call opened?")),

    _S("net_switch_outage", "network", "none", False, 3,
       ("dept",),
       ("an entire floor has lost network connectivity",
        "the switch in the third-floor comms cupboard appears to be down",
        "all wired ports across two floors are dead",
        "no device on that segment can reach anything on the network"),
       ("the uplink lights on the cabinet are off entirely",
        "power to the cupboard looks fine, the UPS panel is normal",
        "wifi in the same area is also affected because the APs are on that switch",
        "it went down without warning at around 11:15"),
       ("please dispatch someone to the comms room",
        "we need this restored urgently",
        "can we get an ETA?",
        "escalating on behalf of the floor")),

    _S("net_firewall_block", "network", "none", False, 2,
       ("small", "team"),
       ("an internal application is being blocked by the firewall after the rule change",
        "we cannot reach the supplier portal from the office network any more",
        "outbound connections on the port our tool needs are being dropped",
        "the integration endpoint is unreachable from inside the network only"),
       ("the same URL works fine from a mobile connection outside the office",
        "it broke at the same time as last night's rule review",
        "no error is returned, the connection simply times out",
        "the vendor confirms their end is up and accepting traffic"),
       ("please review the firewall rules",
        "can an exception be raised?",
        "we need the previous rule restored",
        "who approves changes like this?")),

    _S("net_guest_wifi_captive", "network", "wifi", False, 0,
       ("single", "small"),
       ("the guest wifi login page never appears when visitors connect",
        "guests connect to the visitor network but cannot get past the splash screen",
        "the captive portal shows a certificate warning to every visitor",
        "the guest wifi voucher codes are being rejected at the portal"),
       ("the staff network works normally from the same spot",
        "we have visitors arriving for a workshop tomorrow morning",
        "it affects both iOS and Android devices",
        "forgetting the network and rejoining does not help"),
       ("could you check the portal before the workshop?",
        "please issue fresh voucher codes if needed",
        "any workaround we can give visitors?",
        "let me know if reception needs to do anything")),

    # ---------------- software ----------------
    _S("sw_crm_crash", "software", "crm", False, 2,
       ("small", "team"),
       ("the CRM desktop client crashes whenever a record is saved",
        "the CRM closes without warning when the reports tab is opened",
        "saving an opportunity in the CRM throws an unhandled exception",
        "the CRM freezes and then exits as soon as an attachment is added"),
       ("the web version of the same system behaves correctly",
        "it began after the client was pushed out on Tuesday",
        "a repair install did not change the behaviour",
        "unsaved work is lost each time, which is the painful part"),
       ("please raise it with the vendor",
        "can the previous version be redeployed?",
        "is there a workaround while this is fixed?",
        "we need this resolved before the quarter close")),

    _S("sw_license_expired", "software", "none", False, 1,
       ("small", "team"),
       ("the design software reports that our licence has expired",
        "the statistics package refuses to launch, citing an invalid licence",
        "our licence server is handing out no seats and everyone is locked out of the tool",
        "the application starts in read-only mode because the licence lapsed"),
       ("the renewal was supposedly processed by procurement last month",
        "the licence file on the server has yesterday's date as its expiry",
        "we have a deliverable due at the end of the week",
        "restarting the licence service did not release any seats"),
       ("can procurement confirm the renewal?",
        "please reinstate the licence",
        "is there a temporary seat available?",
        "who owns this contract?")),

    _S("sw_install_request", "software", "none", False, 0,
       ("single", "small"),
       ("I need a diagramming tool installed on my machine",
        "please install the PDF editor that the rest of the team uses",
        "I require the database client tools for my new project",
        "could the screen recording software be added to my laptop?"),
       ("my manager has approved the request in the tooling channel",
        "it is on the approved software list already",
        "I do not have local admin rights to install it myself",
        "the free version is sufficient for what I need"),
       ("could you push it via the software centre?",
        "please let me know when it is available",
        "happy to book a slot for the install",
        "what approval reference do you need?")),

    _S("sw_update_broke_app", "software", "laptop", False, 2,
       ("team", "dept"),
       ("the overnight update has broken the accounting application on every machine",
        "since the patch was applied the finance tool refuses to open",
        "the Windows update rolled out last night stops our main application launching",
        "the update has replaced a component the line-of-business app depends on"),
       ("rolling back the update on one test machine restored the application",
        "the error mentions a missing runtime library",
        "machines that have not yet updated are working normally",
        "this is affecting people who need to close the month"),
       ("please pause the rollout",
        "can the patch be rolled back fleet-wide?",
        "we need guidance for affected staff",
        "escalating as it is blocking work")),

    _S("sw_db_query_timeout", "software", "database", False, 2,
       ("small", "team"),
       ("reporting queries against the warehouse are timing out",
        "the database returns a timeout on anything touching the orders table",
        "queries that used to take seconds now run for minutes and fail",
        "the reporting database rejects connections during business hours"),
       ("the same query against the staging copy returns immediately",
        "it degraded gradually over the past week",
        "the maintenance job may not have completed on Sunday",
        "we can see high wait times but do not have access to investigate"),
       ("could a DBA take a look?",
        "please check whether index maintenance ran",
        "is there a long-running job blocking things?",
        "we need reports out by Friday")),

    _S("sw_payroll_calc_error", "software", "payroll", False, 3,
       ("team", "dept"),
       ("the payroll system is calculating incorrect deductions this cycle",
        "overtime hours are not being picked up by the payroll run",
        "the payroll preview shows the wrong tax band for a large group of staff",
        "the payroll export contains duplicated lines for many employees"),
       ("the pay run is scheduled to be submitted the day after tomorrow",
        "last month's run was correct with the same inputs",
        "the discrepancy appears only for staff who changed contract this year",
        "finance has paused the approval until this is understood"),
       ("we need this checked before the run is submitted",
        "please engage the vendor support line",
        "can someone verify the calculation rules?",
        "escalating given the deadline")),

    _S("sw_file_corrupt", "software", "none", False, 1,
       ("single", "small"),
       ("a spreadsheet on the shared drive will no longer open and reports corruption",
        "the project file opens but all the formulas have been replaced with errors",
        "our shared workbook now says the format is invalid",
        "a document opens as unreadable characters instead of text"),
       ("the version from three days ago opens correctly",
        "it was last saved late yesterday afternoon",
        "the file size looks about right, so the data may still be there",
        "we do not have a local copy of the latest edits"),
       ("can you restore it from backup?",
        "please recover the most recent working version",
        "is there any way to repair the file?",
        "let me know what the retention window is")),

    # ---------------- security ----------------
    _S("sec_phishing_email", "security", "email", True, 2,
       ("single", "small", "team"),
       ("I received an email pretending to be from the finance director asking for a transfer",
        "a message claiming our mailbox is full is asking people to re-enter their password",
        "an email with a fake invoice link has landed in several inboxes",
        "a convincing message impersonating our supplier is asking us to update bank details"),
       ("I have not clicked the link and have kept the message for you",
        "the sender domain is a lookalike with one letter changed",
        "the message arrived well outside normal business hours",
        "the reply-to address is different from the visible sender"),
       ("please confirm whether anyone clicked it",
        "can this be blocked at the gateway?",
        "should I forward it to the security mailbox?",
        "advise what to tell the team")),

    _S("sec_malware_detected", "security", "laptop", True, 3,
       ("single", "small"),
       ("the endpoint agent has flagged a trojan on my machine and quarantined a file",
        "antivirus is repeatedly detecting and re-detecting something in a temp folder",
        "the security tool reports an active threat that it cannot fully remove",
        "a malicious script was blocked but the alert keeps firing"),
       ("I disconnected from the network as soon as the alert appeared",
        "I had opened an attachment from an unknown sender shortly before",
        "the machine is noticeably slower than usual",
        "the alert names a file in my downloads folder"),
       ("please advise whether to keep it offline",
        "can someone image the machine?",
        "what should I do with the device now?",
        "treat as urgent please")),

    _S("sec_data_exfil_alert", "security", "database", True, 3,
       ("team", "dept"),
       ("monitoring flagged an unusually large export from the customer database overnight",
        "a bulk download of records was detected outside normal working hours",
        "the DLP system alerted on a large volume of records leaving the database",
        "an automated alert reports thousands of rows queried by a service account"),
       ("the account involved does not normally run bulk exports",
        "the activity took place between 02:00 and 03:00",
        "no change request covers work at that time",
        "the destination address is not one we recognise"),
       ("please start an investigation now",
        "can the account be suspended pending review?",
        "we need the audit logs preserved",
        "who owns the incident response process?")),

    _S("sec_credential_leak", "security", "none", True, 3,
       ("team", "dept"),
       ("company credentials have appeared in a public breach dump",
        "a third-party notification says our staff email addresses and passwords are circulating",
        "we have been alerted that corporate logins are for sale on a forum",
        "a breach notification service flagged reused passwords from our domain"),
       ("the sample provided matches our email address format",
        "some of the passwords look like current ones rather than historic",
        "the notification arrived through the security mailbox this morning",
        "we have not yet told anyone outside the immediate team"),
       ("we need a forced password reset considered",
        "please assess the scope urgently",
        "can you confirm which accounts are affected?",
        "advise on communications")),

    _S("sec_suspicious_login", "security", "none", True, 2,
       ("single", "small"),
       ("there are sign-ins to my account from a country I have never visited",
        "the security log shows successful logins at times I was asleep",
        "I received MFA prompts I did not trigger, several in a row",
        "an unfamiliar device appears in my account's active sessions"),
       ("I did not approve any of the prompts",
        "the locations listed are thousands of miles apart within an hour",
        "my password has not been changed recently",
        "I have not used any public wifi lately"),
       ("please review the sign-in logs",
        "should I change my password now?",
        "can you kill the other sessions?",
        "let me know what else to check")),

    _S("sec_usb_unknown_device", "security", "laptop", True, 1,
       ("single",),
       ("I found an unbranded USB stick on my desk and have not plugged it in",
        "an unknown USB device was left in the meeting room after an external visit",
        "someone handed me a memory stick from an unidentified source",
        "an unfamiliar USB drive was posted to the office addressed to our team"),
       ("it is sealed in an envelope on my desk for now",
        "nobody in the team claims to have left it",
        "there is no label or identifying mark on it",
        "I know not to insert it, hence the ticket"),
       ("how should I dispose of it?",
        "do you want to examine it?",
        "please advise on the correct process",
        "who should I hand it to?")),

    _S("sec_ransomware_note", "security", "none", True, 3,
       ("team", "dept"),
       ("files on a shared drive have been renamed with an unknown extension and there is a ransom note",
        "a text file demanding payment has appeared in several network folders",
        "documents across a department share will not open and a ransom message is present",
        "a note claiming our files are encrypted has been left in the root of the share"),
       ("we have disconnected the affected share as a precaution",
        "the timestamps suggest it began in the early hours",
        "backups appear to be intact but we have not verified a restore",
        "the note names a contact address and a deadline"),
       ("we need incident response engaged immediately",
        "please advise before anyone touches the files",
        "can you confirm backup integrity?",
        "this is our highest concern right now")),

    # ---------------- account_billing ----------------
    _S("ab_invoice_dispute", "account_billing", "none", False, 1,
       ("single", "small"),
       ("the invoice we received does not match the quoted amount",
        "we have been billed twice for the same service this quarter",
        "the invoice includes line items for services we cancelled",
        "the amount charged is significantly higher than the agreed rate"),
       ("I have the original quote reference and can share it",
        "finance has put the payment on hold pending clarification",
        "the difference is around fifteen percent",
        "the supplier has not responded to two emails"),
       ("can you help query this with the vendor?",
        "please advise who owns this contract",
        "what is the process for disputing an invoice?",
        "let me know what evidence you need")),

    _S("ab_license_overcharge", "account_billing", "none", False, 1,
       ("team", "dept"),
       ("we are being charged for more software seats than we actually use",
        "the subscription bill lists licences for people who have left",
        "our seat count on the invoice is well above the number of active users",
        "we appear to be paying for a tier we do not need"),
       ("a quick count suggests a large number of unused seats",
        "the renewal is due at the end of the month",
        "nobody seems to own the licence reconciliation",
        "the vendor portal shows a different count from the invoice"),
       ("can we reconcile the seat count before renewal?",
        "please advise how to reduce the tier",
        "who can approve a downgrade?",
        "we would like this addressed this cycle")),

    _S("ab_subscription_renewal", "account_billing", "none", False, 1,
       ("small", "team"),
       ("our team subscription lapses at the end of next week",
        "the renewal for the collaboration tool has not been actioned",
        "the annual subscription is due and no purchase order exists yet",
        "we received a final reminder that the plan will downgrade automatically"),
       ("losing it would stop the team sharing work externally",
        "the budget was approved in the last planning round",
        "the cost is unchanged from last year",
        "procurement asked us to raise this through the service desk"),
       ("can you start the renewal process?",
        "please confirm the purchase order number",
        "who needs to sign this off?",
        "let me know the lead time")),

    _S("ab_payroll_deduction_q", "account_billing", "payroll", False, 1,
       ("single", "small"),
       ("there is a deduction on my payslip that I do not recognise",
        "my payslip shows a benefit charge I never signed up for",
        "the pension contribution on my payslip is not the percentage I selected",
        "an expenses recovery line has appeared on my pay without explanation"),
       ("the amount is small but it has appeared for three months running",
        "I checked my benefits portal and it shows the correct selection",
        "HR suggested raising it here first",
        "I have the payslip reference to hand"),
       ("could someone explain the deduction?",
        "please advise who I should speak to",
        "can this be corrected in the next run?",
        "happy to send the payslip over")),

    _S("ab_expense_reimbursement", "account_billing", "none", False, 0,
       ("single",),
       ("my expense claim from six weeks ago has still not been reimbursed",
        "the expenses system shows my claim as approved but nothing has been paid",
        "a travel claim appears to have been lost in the approval workflow",
        "I cannot submit receipts because the expenses tool rejects the upload"),
       ("my manager confirms they approved it at the time",
        "the claim reference is visible in the portal",
        "the amount is a few hundred pounds of train fares",
        "I have digital copies of every receipt"),
       ("could you chase this for me?",
        "please advise who handles expense queries",
        "is there a known backlog?",
        "let me know if I should resubmit")),

    _S("ab_vendor_quote", "account_billing", "none", False, 0,
       ("single", "small"),
       ("we need a quote for additional storage on our hosting plan",
        "please obtain pricing for extending our support contract",
        "we would like costings for adding two more seats to the analytics tool",
        "can we get a quote for replacement monitors for the team?"),
       ("this is for next year's budget planning rather than an urgent purchase",
        "the finance business partner has asked for indicative numbers",
        "we need it before the planning meeting in two weeks",
        "no commitment is being made at this stage"),
       ("could you request it from the supplier?",
        "please share the quote when it arrives",
        "who is our account manager there?",
        "let me know if you need a specification")),

    _S("ab_cost_center_wrong", "account_billing", "none", False, 0,
       ("small", "team"),
       ("our software charges are being posted to the wrong cost centre",
        "the IT recharge has landed against a department that did not order it",
        "the internal billing code on our services is out of date",
        "charges for our tooling are appearing in another team's budget"),
       ("the reorganisation in April changed our reporting line",
        "finance flagged it during the month-end review",
        "we have the correct code and can supply it",
        "it has been wrong for at least two months"),
       ("can the code be corrected going forward?",
        "please advise whether past months can be reallocated",
        "who updates the recharge mapping?",
        "happy to confirm the new code in writing")),

    # ---------------- email ----------------
    _S("email_not_receiving", "email", "email", False, 2,
       ("small", "team"),
       ("external emails are not arriving in our mailboxes",
        "messages from customers are not being delivered to the shared inbox",
        "we have received nothing from outside the company since this morning",
        "inbound mail appears to be silently disappearing"),
       ("internal mail between colleagues works normally",
        "senders are not getting any bounce message back",
        "the shared inbox is the main channel for customer requests",
        "a test from a personal account also failed to arrive"),
       ("please check the mail flow rules",
        "can someone look at the message trace?",
        "we are missing customer requests",
        "we need an update quickly")),

    _S("email_mailbox_full", "email", "email", False, 1,
       ("single", "small"),
       ("my mailbox has hit its size limit and I can no longer send",
        "I am getting quota warnings and outgoing mail is being held",
        "the mailbox is full and archiving does not seem to free any space",
        "I cannot send attachments because the mailbox is over quota"),
       ("I have already emptied deleted items and the junk folder",
        "the archive folder does not appear to be reducing the total",
        "most of the size is old attachments from a project that finished",
        "receiving still works, it is only sending that is blocked"),
       ("could my quota be increased?",
        "please advise on archiving options",
        "is there a retention policy I should apply?",
        "let me know the best way to reduce it")),

    _S("email_calendar_sync", "email", "phone", False, 1,
       ("single", "small"),
       ("my calendar on the phone is days out of date compared with the desktop",
        "meetings accepted on my laptop never appear on my mobile calendar",
        "the phone shows meetings that were cancelled last week",
        "calendar sync on the handset stopped after the operating system update"),
       ("email itself syncs fine on the same device, only the calendar lags",
        "removing and re-adding the account did not fix it",
        "I nearly missed a client meeting because of this",
        "the handset was replaced about two months ago"),
       ("can someone look at the mobile profile?",
        "please advise on a reset procedure",
        "is there a known issue with this version?",
        "happy to bring the phone in")),

    _S("email_dl_update", "email", "none", False, 0,
       ("small", "team"),
       ("please add three new starters to the department distribution list",
        "the team distribution list still contains people who have moved on",
        "we need a new distribution list creating for the project group",
        "the mailing list owner has left and nobody can edit it now"),
       ("I can supply the full list of names and addresses",
        "the manager has approved the membership changes",
        "it is used for weekly operational updates",
        "there is no urgency, but we would like it before the next cycle"),
       ("could you update the membership?",
        "please confirm once the list is changed",
        "who can be set as the new owner?",
        "let me know if you need approval in writing")),

    _S("email_spam_flood", "email", "email", False, 1,
       ("team", "dept"),
       ("staff are receiving large volumes of unsolicited marketing mail",
        "junk mail has increased sharply across the office this week",
        "the spam filter appears to be letting far more through than usual",
        "inboxes are filling with bulk mail that used to be blocked"),
       ("legitimate messages are getting lost among the noise",
        "it began around the start of the week",
        "the messages come from many different domains",
        "marking as junk does not seem to be having an effect"),
       ("can the filtering thresholds be reviewed?",
        "please check whether a policy changed",
        "is there anything staff should do differently?",
        "we would like this brought back under control")),

    _S("email_spoofed_sender", "email", "email", True, 2,
       ("team", "dept"),
       ("messages are going out that appear to come from our domain but were not sent by us",
        "customers are reporting emails from our address that we never sent",
        "our domain is being spoofed in messages sent to external parties",
        "bounce messages are arriving for mail we did not originate"),
       ("the message headers show an unrelated sending server",
        "the content asks recipients to update payment details",
        "several customers have contacted us to check",
        "our own mailboxes show no trace of the messages in sent items"),
       ("please review the domain authentication records",
        "can we tighten the policy to stop this?",
        "we need advice on what to tell customers",
        "can someone action this today?")),

    _S("email_signature_broken", "email", "none", False, 0,
       ("small", "team"),
       ("the corporate email signature is not being applied to outgoing messages",
        "our signatures show the old company logo and address",
        "the signature block appears as broken image placeholders for recipients",
        "the disclaimer text has stopped appearing at the bottom of emails"),
       ("it looks correct when composing but wrong when received",
        "it changed after the branding update went out",
        "the mobile app shows a different signature again",
        "it is cosmetic but customer-facing"),
       ("could the template be checked?",
        "please advise how to update it",
        "who maintains the signature policy?",
        "let me know when it is corrected")),

    # ---------------- other ----------------
    _S("oth_onboarding_setup", "other", "none", False, 1,
       ("single", "small"),
       ("a new starter joins on Monday and nothing has been set up for them",
        "we need a full workstation and accounts prepared for an incoming hire",
        "a graduate starts next week and no equipment has been allocated",
        "the onboarding request for a new joiner does not seem to have been picked up"),
       ("HR submitted the joiner form ten days ago",
        "they will be based in the London office on the second floor",
        "their induction schedule starts at nine on their first morning",
        "we can supply the cost centre and manager details"),
       ("can you confirm what is still outstanding?",
        "please advise on the lead time",
        "what do you need from the hiring manager?",
        "we would like everything ready before they arrive")),

    _S("oth_meeting_room_av", "other", "none", False, 1,
       ("small", "team"),
       ("the screen in the large meeting room will not display anything from a laptop",
        "the conference room camera is not detected by the meeting software",
        "audio in the boardroom cuts out repeatedly during calls",
        "the room booking panel outside the meeting room is frozen"),
       ("we have tried both the HDMI cable and the wireless option",
        "there is a client workshop booked in that room tomorrow",
        "the smaller rooms on the same floor work fine",
        "the equipment was serviced two months ago"),
       ("could someone check it before tomorrow?",
        "please advise on an alternative room",
        "is there a spare kit we can borrow?",
        "let me know if it needs a supplier visit")),

    _S("oth_desk_move", "other", "none", False, 0,
       ("small", "team"),
       ("we are moving desks next week and need the equipment relocated",
        "our team is relocating to the fourth floor and will need machines moved",
        "please arrange for monitors and docks to be moved to the new seating area",
        "the desk move on Friday needs IT to disconnect and reconnect equipment"),
       ("facilities have confirmed the new floor plan",
        "we would like to keep downtime to a minimum on the day",
        "the move is scheduled for after four in the afternoon",
        "the seating chart has been shared with the office manager"),
       ("can you schedule someone for the move?",
        "please confirm what we should pack ourselves",
        "who coordinates this with facilities?",
        "let us know the timings")),

    _S("oth_badge_access", "other", "none", False, 1,
       ("single", "small"),
       ("my building badge no longer opens the third-floor door",
        "the access card for a new joiner does not work on any reader",
        "my pass works at the main entrance but not in the secure area",
        "the badge reader rejects my card with a red light every time"),
       ("it worked last week without any issue",
        "the card is not visibly damaged or bent",
        "reception could not resolve it and suggested raising a ticket",
        "I need access for the equipment audit on Thursday"),
       ("could my permissions be checked?",
        "please advise whether a new card is needed",
        "who manages door access groups?",
        "happy to come to reception")),

    _S("oth_training_request", "other", "none", False, 0,
       ("single", "small"),
       ("I would like access to the internal training platform",
        "please enrol me on the data protection refresher course",
        "our team needs the security awareness module assigned",
        "I cannot find the course my manager asked me to complete"),
       ("my manager has approved the time for this",
        "the deadline for completion is the end of the quarter",
        "I completed the previous version last year",
        "the platform does not list it under my assigned learning"),
       ("could you assign the course?",
        "please advise how enrolment works",
        "who administers the learning platform?",
        "let me know once it appears")),

    _S("oth_asset_inventory", "other", "none", False, 0,
       ("small", "team"),
       ("we need an inventory list of all equipment assigned to our team",
        "please confirm which assets are recorded against our cost centre",
        "we are preparing for an audit and need the asset register extract",
        "the equipment list we hold does not match what people actually have"),
       ("the audit is scheduled for the end of next month",
        "we have found several items with no asset tag",
        "finance has asked for this in a specific format",
        "we can help verify the list once we have it"),
       ("could you export the register for us?",
        "please advise on the tagging process",
        "who maintains the asset database?",
        "let me know a realistic turnaround")),

    _S("oth_server_room_ac", "other", "none", False, 3,
       ("dept",),
       ("the air conditioning in the server room has failed and temperatures are climbing",
        "the comms room is significantly hotter than normal and the cooling unit is silent",
        "a temperature alarm is sounding in the equipment room",
        "the cooling in the data cabinet area has stopped and kit is running hot"),
       ("the thermometer on the wall is well above the normal range",
        "we have propped the door open as a temporary measure",
        "equipment has not shut down yet but it is getting warmer",
        "the alarm panel shows a fault on the cooling unit"),
       ("please get facilities and IT on this now",
        "we risk equipment shutting down",
        "can someone attend immediately?",
        "escalating -- this affects everything hosted here")),
)


# --------------------------------------------------------------------------
# Template families (surface style only -- carry no label information)
# --------------------------------------------------------------------------

TEMPLATE_FAMILIES: tuple[str, ...] = (
    "terse",
    "polite_request",
    "frustrated",
    "formal_ticket",
    "chat_style",
    "manager_escalation",
    "narrative",
    "bulleted",
)

_OPENERS_POLITE = ("Hi IT team,", "Hello service desk,", "Morning all,", "Hi there,")
_SIGNOFFS_POLITE = ("Thanks in advance.", "Many thanks.", "Thanks for your help.",
                    "Appreciate it.")
_FRUSTRATION = (
    "This is the third time this week and it is really holding us up.",
    "I have already raised this once and heard nothing back.",
    "We are losing hours to this and it needs sorting.",
    "Honestly this has been going on far too long.",
)
_ESCALATION = (
    "Escalating on behalf of the team.",
    "Raising this at the request of the department head.",
    "Flagging this formally as it is now affecting delivery.",
    "Please treat this with appropriate urgency.",
)


def _cap(text: str) -> str:
    """Upper-case the first character only.

    NOT str.capitalize(), which lower-cases everything after the first
    character and would turn "the CRM" into "the crm" and "I try" into "i try".
    That defect was found in manual review round 1; see DATASET_CARD.md.
    """
    return text[:1].upper() + text[1:] if text else text


def _who_phrase(users: int, r) -> str:
    """Render the number of affected people so it is always recoverable."""
    if users == 1:
        return r.choice((
            "It is just me affected.",
            "This is only affecting me as far as I can tell.",
            "I am the only person impacted.",
            "Only one person is affected.",
        ))
    templates = (
        "About {n} people are affected.",
        "This is hitting {n} users so far.",
        "We have {n} staff impacted at the moment.",
        "Around {n} colleagues have reported the same thing.",
        "{n} users are affected in total.",
        "So far {n} people have reported it.",
    )
    return r.choice(templates).format(n=users)


def _typo(text: str, r) -> str:
    """Light, deterministic surface noise for the chat_style template."""
    swaps = {" the ": " teh ", " and ": " an ", "please": "pls",
             "cannot": "cant", " to ": " to ", "because": "becuase"}
    keys = [k for k in swaps if k in text]
    if keys:
        k = r.choice(keys)
        text = text.replace(k, swaps[k], 1)
    return text


def render(template_family: str, sym: str, det: str, ask: str,
           who: str, r) -> str:
    """Render one ticket body in the given surface style."""
    if template_family == "terse":
        return f"{_cap(sym)}. {who} {_cap(ask)}"

    if template_family == "polite_request":
        return (f"{r.choice(_OPENERS_POLITE)} {sym}. {_cap(det)}. "
                f"{who} {_cap(ask)} {r.choice(_SIGNOFFS_POLITE)}")

    if template_family == "frustrated":
        return (f"{_cap(sym)}. {r.choice(_FRUSTRATION)} "
                f"{_cap(det)}. {who} {_cap(ask)}")

    if template_family == "formal_ticket":
        return (f"Issue summary: {sym}. "
                f"Additional detail: {det}. "
                f"Impact: {who} "
                f"Request: {ask}")

    if template_family == "chat_style":
        body = f"hey, {sym}. {det}. {who.lower()} {ask}"
        return _typo(body.lower(), r)

    if template_family == "manager_escalation":
        return (f"{r.choice(_ESCALATION)} {_cap(sym)}. "
                f"{_cap(det)}. {who} {_cap(ask)}")

    if template_family == "narrative":
        return (f"I want to explain what has happened. {_cap(sym)}. "
                f"{_cap(det)}. {who} At this point {ask}")

    if template_family == "bulleted":
        return (f"- Problem: {sym}\n"
                f"- Detail: {det}\n"
                f"- Scope: {who}\n"
                f"- Request: {ask}")

    raise ValueError(f"unknown template family: {template_family}")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

# How many examples each family contributes. 56 families -> 300 examples:
# 20 families produce 6 examples, 36 produce 5. Assignment is by index so it is
# reproducible and does not depend on dict ordering.
BASE_PER_FAMILY = 5
EXTRA_EXAMPLE_FAMILIES = 20
TARGET_TOTAL = 300


def examples_per_family(index: int) -> int:
    return BASE_PER_FAMILY + (1 if index < EXTRA_EXAMPLE_FAMILIES else 0)


def _sample_users(scale: str, r) -> int:
    lo, hi = USER_SCALE_RANGES[scale]
    return r.randint(lo, hi)


def generate_dataset() -> list[dict[str, Any]]:
    """Build the full dataset deterministically.

    Running this twice on any machine produces byte-identical output; that
    property is asserted in the test suite and the file checksum is recorded in
    the run ledger.
    """
    records: list[dict[str, Any]] = []

    for fam_index, fam in enumerate(FAMILIES):
        n = examples_per_family(fam_index)
        for k in range(n):
            r = rng("dataset_generation", fam.fid, k)

            template_family = TEMPLATE_FAMILIES[
                (fam_index + k) % len(TEMPLATE_FAMILIES)
            ]
            scale = r.choice(fam.user_scales)
            users = _sample_users(scale, r)

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
                # Clause-level lineage: which building blocks produced this
                # text. Makes every example auditable back to its source
                # fragments without re-running the generator.
                "symptom_text": sym,
                "detail_text": det,
                "ask_text": ask,
                "scope_text": who,
                "source": "synthetic:forgelm.datagen",
                "generator_version": DATASET_VERSION,
                "schema_version": "1.0.0",
                "generation_seed": SEEDS["dataset_generation"],
                "qc_status": "pending",
            })

    return records


def write_dataset(path: str | Path) -> tuple[Path, int]:
    """Write the dataset as JSON Lines and return (path, count)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_dataset()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path, len(records)


def family_table() -> list[dict[str, Any]]:
    """Flat description of every scenario family, for the dataset card."""
    return [
        {**asdict(f),
         "symptoms": len(f.symptoms),
         "details": len(f.details),
         "asks": len(f.asks),
         "n_examples": examples_per_family(i)}
        for i, f in enumerate(FAMILIES)
    ]
