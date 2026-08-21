# Manual review sample

One example per (split, category) -- 24 in total. Deterministic: the same 24 examples every time, so a second reviewer reads exactly what the first one did.

Regenerate with `python scripts/05_review_sample.py`.

## What to check when reading

1. Does the ticket read like something a person would actually write?
2. Is `users_affected` genuinely recoverable from the text?
3. Do the clauses contradict each other or the label?
4. Does the text give away `priority` directly rather than implying it?
5. Are acronyms (CRM, VPN, MFA, SSO, DNS) correctly cased?

Defects found in round 1 and since fixed are listed in `DATASET_CARD.md` under *Manual review*.

## train (n=171)

### access_management -- `am_ex_employee_active-00`

- style: `narrative` | scenario: `am_ex_employee_active` | base_severity: 3 | user_scale: `single`

```
I want to explain what has happened. We found sign-in activity on a disabled-user account that should not exist. No ticket exists explaining why the account was re-enabled. It is just me affected. At this point can you pull the full sign-in history?
```

Expected: `{"category": "access_management", "priority": "high", "affected_service": "none", "is_security_incident": true, "users_affected": 1}`

### hardware -- `hw_battery_swelling-00`

- style: `terse` | scenario: `hw_battery_swelling` | base_severity: 3 | user_scale: `single`

```
The battery compartment is expanding and the chassis is separating. This is only affecting me as far as I can tell. Please arrange safe disposal and a replacement
```

Expected: `{"category": "hardware", "priority": "high", "affected_service": "laptop", "is_security_incident": false, "users_affected": 1}`

### network -- `net_guest_wifi_captive-00`

- style: `chat_style` | scenario: `net_guest_wifi_captive` | base_severity: 0 | user_scale: `single`

```
hey, the captive portal shows a certificate warning to every visitor. forgetting the network an rejoining does not help. it is just me affected. please issue fresh voucher codes if needed
```

Expected: `{"category": "network", "priority": "low", "affected_service": "wifi", "is_security_incident": false, "users_affected": 1}`

### software -- `sw_db_query_timeout-00`

- style: `polite_request` | scenario: `sw_db_query_timeout` | base_severity: 2 | user_scale: `team`

```
Hello service desk, the database returns a timeout on anything touching the orders table. The same query against the staging copy returns immediately. So far 12 people have reported it. We need reports out by Friday Appreciate it.
```

Expected: `{"category": "software", "priority": "high", "affected_service": "database", "is_security_incident": false, "users_affected": 12}`

### security -- `sec_credential_leak-00`

- style: `bulleted` | scenario: `sec_credential_leak` | base_severity: 3 | user_scale: `dept`

```
- Problem: company credentials have appeared in a public breach dump
- Detail: we have not yet told anyone outside the immediate team
- Scope: So far 90 people have reported it.
- Request: please assess the scope urgently
```

Expected: `{"category": "security", "priority": "critical", "affected_service": "none", "is_security_incident": true, "users_affected": 90}`

### account_billing -- `ab_cost_center_wrong-00`

- style: `polite_request` | scenario: `ab_cost_center_wrong` | base_severity: 0 | user_scale: `team`

```
Morning all, the IT recharge has landed against a department that did not order it. We have the correct code and can supply it. This is hitting 15 users so far. Happy to confirm the new code in writing Thanks in advance.
```

Expected: `{"category": "account_billing", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 15}`

### email -- `email_dl_update-00`

- style: `manager_escalation` | scenario: `email_dl_update` | base_severity: 0 | user_scale: `small`

```
Please treat this with appropriate urgency. We need a new distribution list creating for the project group. The manager has approved the membership changes. We have 6 staff impacted at the moment. Who can be set as the new owner?
```

Expected: `{"category": "email", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 6}`

### other -- `oth_desk_move-00`

- style: `formal_ticket` | scenario: `oth_desk_move` | base_severity: 0 | user_scale: `small`

```
Issue summary: the desk move on Friday needs IT to disconnect and reconnect equipment. Additional detail: we would like to keep downtime to a minimum on the day. Impact: So far 7 people have reported it. Request: can you schedule someone for the move?
```

Expected: `{"category": "other", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 7}`

## validation (n=43)

### access_management -- `am_mfa_device_lost-00`

- style: `polite_request` | scenario: `am_mfa_device_lost` | base_severity: 1 | user_scale: `single`

```
Hi there, I lost the phone that had my authenticator app on it. I still have my password, it is only the second factor that is blocking me. This is only affecting me as far as I can tell. What is the process to reset this? Thanks for your help.
```

Expected: `{"category": "access_management", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 1}`

### hardware -- `hw_laptop_wont_boot-00`

- style: `bulleted` | scenario: `hw_laptop_wont_boot` | base_severity: 2 | user_scale: `small`

```
- Problem: my laptop powers on but never gets past the manufacturer logo
- Detail: it was working normally when I shut it down last night
- Scope: This is hitting 4 users so far.
- Request: please advise where to bring it
```

Expected: `{"category": "hardware", "priority": "medium", "affected_service": "laptop", "is_security_incident": false, "users_affected": 4}`

### network -- `net_firewall_block-00`

- style: `formal_ticket` | scenario: `net_firewall_block` | base_severity: 2 | user_scale: `team`

```
Issue summary: we cannot reach the supplier portal from the office network any more. Additional detail: it broke at the same time as last night's rule review. Impact: This is hitting 34 users so far. Request: please review the firewall rules
```

Expected: `{"category": "network", "priority": "high", "affected_service": "none", "is_security_incident": false, "users_affected": 34}`

### software -- `sw_crm_crash-00`

- style: `manager_escalation` | scenario: `sw_crm_crash` | base_severity: 2 | user_scale: `small`

```
Raising this at the request of the department head. Saving an opportunity in the CRM throws an unhandled exception. A repair install did not change the behaviour. We have 2 staff impacted at the moment. We need this resolved before the quarter close
```

Expected: `{"category": "software", "priority": "medium", "affected_service": "crm", "is_security_incident": false, "users_affected": 2}`

### security -- `sec_ransomware_note-00`

- style: `frustrated` | scenario: `sec_ransomware_note` | base_severity: 3 | user_scale: `team`

```
A note claiming our files are encrypted has been left in the root of the share. We are losing hours to this and it needs sorting. The note names a contact address and a deadline. We have 31 staff impacted at the moment. Can you confirm backup integrity?
```

Expected: `{"category": "security", "priority": "critical", "affected_service": "none", "is_security_incident": true, "users_affected": 31}`

### account_billing -- `ab_license_overcharge-00`

- style: `chat_style` | scenario: `ab_license_overcharge` | base_severity: 1 | user_scale: `dept`

```
hey, we are being charged for more software seats than we actually use. nobody seems to own teh licence reconciliation. this is hitting 70 users so far. can we reconcile the seat count before renewal?
```

Expected: `{"category": "account_billing", "priority": "high", "affected_service": "none", "is_security_incident": false, "users_affected": 70}`

### email -- `email_spam_flood-00`

- style: `narrative` | scenario: `email_spam_flood` | base_severity: 1 | user_scale: `dept`

```
I want to explain what has happened. Junk mail has increased sharply across the office this week. Legitimate messages are getting lost among the noise. Around 155 colleagues have reported the same thing. At this point we would like this brought back under control
```

Expected: `{"category": "email", "priority": "high", "affected_service": "email", "is_security_incident": false, "users_affected": 155}`

### other -- `oth_badge_access-00`

- style: `chat_style` | scenario: `oth_badge_access` | base_severity: 1 | user_scale: `small`

```
hey, my building badge no longer opens the third-floor door. reception could not resolve it an suggested raising a ticket. this is hitting 5 users so far. could my permissions be checked?
```

Expected: `{"category": "other", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 5}`

## test (n=86)

### access_management -- `am_account_lockout-00`

- style: `frustrated` | scenario: `am_account_lockout` | base_severity: 1 | user_scale: `small`

```
Something on the network appears to be retrying an old password. Honestly this has been going on far too long. It unlocks fine, then locks again within about five minutes. So far 3 people have reported it. Help would be appreciated
```

Expected: `{"category": "access_management", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 3}`

### hardware -- `hw_monitor_dead-00`

- style: `formal_ticket` | scenario: `hw_monitor_dead` | base_severity: 1 | user_scale: `small`

```
Issue summary: one of the desk monitors shows vertical coloured lines across the panel. Additional detail: the same monitor fails on a colleague's laptop as well. Impact: About 4 people are affected. Request: can it be replaced from stock?
```

Expected: `{"category": "hardware", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 4}`

### network -- `net_dns_failure-00`

- style: `polite_request` | scenario: `net_dns_failure` | base_severity: 3 | user_scale: `dept`

```
Hi there, internal name resolution has failed and nothing resolves by hostname. It began immediately after the overnight change window. 148 users are affected in total. This is stopping work across the site Thanks in advance.
```

Expected: `{"category": "network", "priority": "critical", "affected_service": "none", "is_security_incident": false, "users_affected": 148}`

### software -- `sw_file_corrupt-00`

- style: `formal_ticket` | scenario: `sw_file_corrupt` | base_severity: 1 | user_scale: `small`

```
Issue summary: a document opens as unreadable characters instead of text. Additional detail: the file size looks about right, so the data may still be there. Impact: So far 8 people have reported it. Request: is there any way to repair the file?
```

Expected: `{"category": "software", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 8}`

### security -- `sec_malware_detected-00`

- style: `manager_escalation` | scenario: `sec_malware_detected` | base_severity: 3 | user_scale: `small`

```
Flagging this formally as it is now affecting delivery. The endpoint agent has flagged a trojan on my machine and quarantined a file. I disconnected from the network as soon as the alert appeared. About 4 people are affected. Please advise whether to keep it offline
```

Expected: `{"category": "security", "priority": "high", "affected_service": "laptop", "is_security_incident": true, "users_affected": 4}`

### account_billing -- `ab_expense_reimbursement-00`

- style: `bulleted` | scenario: `ab_expense_reimbursement` | base_severity: 0 | user_scale: `single`

```
- Problem: my expense claim from six weeks ago has still not been reimbursed
- Detail: my manager confirms they approved it at the time
- Scope: It is just me affected.
- Request: is there a known backlog?
```

Expected: `{"category": "account_billing", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 1}`

### email -- `email_calendar_sync-00`

- style: `chat_style` | scenario: `email_calendar_sync` | base_severity: 1 | user_scale: `single`

```
hey, meetings accepted on my laptop never appear on my mobile calendar. the handset was replaced about two months ago. it is just me affected. pls advise on a reset procedure
```

Expected: `{"category": "email", "priority": "low", "affected_service": "phone", "is_security_incident": false, "users_affected": 1}`

### other -- `oth_asset_inventory-00`

- style: `narrative` | scenario: `oth_asset_inventory` | base_severity: 0 | user_scale: `small`

```
I want to explain what has happened. Please confirm which assets are recorded against our cost centre. We can help verify the list once we have it. So far 8 people have reported it. At this point let me know a realistic turnaround
```

Expected: `{"category": "other", "priority": "low", "affected_service": "none", "is_security_incident": false, "users_affected": 8}`
