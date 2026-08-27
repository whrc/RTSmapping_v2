# `ARCHITECTURE` — the control node

Setting up the office PC (Windows) as the machine you drive everything from.

> **The rule this doc exists to enforce:** `ARCHITECTURE` is where you *type*. It is never
> where unattended work *runs*. Anything that must survive you closing your laptop —
> acquisition loops, export drivers, cron alerters — belongs on `rts-ops`. See
> [README.md §2](README.md).

Machine inventory: [README.md](README.md). Projects and buckets:
[infrastructure.md](infrastructure.md).

---

## 1. Prerequisites (once)

**Done 2026-08-27** — this section is now a record of how the box is configured, not a
to-do list. Re-read it when rebuilding the machine.

1. **Google Cloud SDK** — <https://cloud.google.com/sdk/docs/install>. Then, in PowerShell:

   ```powershell
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project abruptthawmapping
   gcloud config set compute/zone us-west1-a
   ```

   Two logins, not one. `auth login` is you at the CLI; `application-default login` writes
   the credential that Python libraries (`google-cloud-storage`, `earthengine-api`) pick up.

   **They are deliberately two different identities on this box**, and the split matters:

   | Layer | Identity | Holds | Used for |
   |---|---|---|---|
   | `gcloud` CLI | `rtsmapping@woodwellclimate.org` | `roles/editor` | provisioning ([pdg_migration.md §4b](pdg_migration.md)), VM lifecycle, SSH |
   | ADC (Python libs) | `yyang@woodwellclimate.org` | `viewer` + `storage.objectUser` + `earthengine.admin` | reading PDG buckets, the bulk copy, Earth Engine |

   `yyang@` cannot provision (it is `viewer`) and `rtsmapping@` is not the Earth Engine
   admin. Do not "tidy" them into one account. If `gcloud config list` warns that the
   active project does not match the ADC quota project, fix the *quota project* — not the
   identity: `gcloud auth application-default set-quota-project abruptthawmapping`.

   Verified on this box: SDK 565.0.0 · CLI `rtsmapping@` · ADC `yyang@` (cloud-platform
   scope) · project `abruptthawmapping` · zone `us-west1-a` · region `us-west1`. The ADC
   reads all four PDG source buckets (`pdg-planet-data`, `rts-mapping-v2`,
   `rts-mapping-v2-usw1`, `rts-mapping-v2-usc1`) — the §4b step-6 precondition, checked
   ahead of the copy.

2. **VSCode** with the **Remote - SSH** extension. *(Installed: `ms-vscode-remote.remote-ssh`,
   `-ssh-edit`, `ms-vscode.remote-explorer`.)*

3. **Confirm IAP works** before relying on it:

   ```powershell
   gcloud compute ssh rts-ops --tunnel-through-iap
   ```

   A `4033 not authorized` means the `roles/iap.tunnelResourceAccessor` grant is missing —
   ask the project owner. IAP TCP forwarding also needs the VPC to allow ingress
   from `35.235.240.0/20` on `tcp:22`; the default network's `default-allow-ssh` covers it
   (checked 2026-08-27: `default-allow-ssh` is `0.0.0.0/0 tcp:22`, so no firewall work).

### 1a. IAP access — RESOLVED 2026-08-27, and two traps worth keeping

`gcloud compute ssh rts-ops --tunnel-through-iap` works from `ARCHITECTURE`. What it took, and
the two things that cost time:

**Trap 1 — `roles/editor` does not carry IAP tunnel access; `roles/owner` does.** Editor grants
sixteen `iap.*` permissions, all of them settings/admin surfaces, and `accessViaIAP` is not among
them. So the original state — `rtsmapping@` as editor — could provision `rts-ops` but could never
log into it. Owner *does* include `accessViaIAP`, which means anyone checking this from an owner's
seat sees it working and concludes there is nothing to grant. Confirmed both ways with
`testIamPermissions`, not read off the role docs.

The fix, run once by an owner (`yyang@` holds `roles/owner`):

```powershell
gcloud projects add-iam-policy-binding abruptthawmapping `
    --member=user:rtsmapping@woodwellclimate.org `
    --role=roles/iap.tunnelResourceAccessor `
    --condition=None
```

`rtsmapping@` gets the one narrow role rather than being made a second owner: it is the everyday
SSH identity, and `yyang@`'s owner+ADC is what the bulk copy and every Earth Engine call run under.
Keep them separate.

**Trap 2 — `posixAccounts[0]` is the wrong username here.** The profile holds *two* POSIX accounts,
and which one applies depends on whether the project is inside `woodwellclimate.org`:

| Username | uid | Use for |
|---|---|---|
| `ext_rtsmapping_woodwellclimate_o` | 3608923965 | out-of-org projects — i.e. **PDG** |
| `rtsmapping_woodwellclimate_org` | 943894250 | in-org projects — i.e. **`abruptthawmapping`** |

The `ext_` form is listed first, so the `--format="value(posixAccounts[0].username)"` idiom in §3
hands you the PDG name, and SSH to `rts-ops` fails with `Permission denied (publickey)` — which
reads like a key problem and is not one. The key was registered correctly the whole time. List
both and pick by project:

```powershell
gcloud compute os-login describe-profile --format="value(posixAccounts[].username)"
```

**Reading the failure modes.** They are distinguishable, so do not guess:

| Symptom | Meaning |
|---|---|
| `4033 not authorized` | the IAP role is missing — Trap 1 |
| `Permission denied (publickey)` **from the remote host** | the tunnel worked; wrong username — Trap 2 |
| `FATAL ERROR: Remote side unexpectedly closed` | usually the VM is still booting; wait and retry |
| `OSError: [Errno 5] stdin ReadFile failed` + traceback | cosmetic. gcloud's Windows tunnel teardown when stdin is not a real console. The command already ran — check stdout |

Firewall needed no work: `default-allow-ssh` is `0.0.0.0/0 tcp:22`, which covers the IAP range
`35.235.240.0/20`.

## 2. Daily lifecycle, from the desk

```powershell
# What exists, and what is running?
gcloud compute instances list

# Ops box — the unattended loops live here
gcloud compute instances start rts-ops --zone us-west1-a
gcloud compute instances stop  rts-ops --zone us-west1-a

# Review app — leave running while the rating campaign is open
gcloud compute instances start rts-review --zone us-west1-a
gcloud compute instances stop  rts-review --zone us-west1-a
```

Stopping a VM stops its per-hour charge; the boot disk still bills (a few dollars a month).

## 3. SSH over IAP — no external IP

`rts-ops` has no public address. It is reached through Google's Identity-Aware Proxy, which
is both simpler and safer than the external-IP arrangement it replaces: nothing is exposed to
the internet, and there is no authorized-networks list to re-authorize when your office IP
changes.

```powershell
gcloud compute ssh rts-ops --zone us-west1-a --tunnel-through-iap
```

### VSCode Remote-SSH through the tunnel

**Already written on `ARCHITECTURE`** (2026-08-27, appended to `C:\Users\Yili Yang\.ssh\config`;
the pre-migration file is backed up beside it as `config.bak-premigration`):

```
Host rts-ops
    HostName        rts-ops
    User            rtsmapping_woodwellclimate_org
    IdentityFile    "C:\Users\Yili Yang\.ssh\google_compute_engine"
    ProxyCommand    "C:\Users\Yili Yang\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" compute start-iap-tunnel %h %p --listen-on-stdin --project=abruptthawmapping --zone=us-west1-a --verbosity=warning
    StrictHostKeyChecking no
```

**Check the gcloud path before copying this elsewhere.** `ARCHITECTURE` has a *per-user*
SDK install under `AppData\Local`, not the `C:\Program Files (x86)\Google\Cloud SDK\...`
path most write-ups assume — that path does not exist on this machine, and an SSH config
pointing at it fails with a bare "connection closed" that looks nothing like a path error.
Get the real one from `(Get-Command gcloud).Source`.

Then **Ctrl+Shift+P → Remote-SSH: Connect to Host → `rts-ops`**. The username above is the
**in-org** one; `posixAccounts[0]` returns the `ext_` PDG name instead — see §1a Trap 2 before
changing it.

The `google_compute_engine` key pair is created for you the first time you run
`gcloud compute ssh`. It already exists here (`~/.ssh/google_compute_engine`, Feb 2026).

**Verified working 2026-08-27** — `ssh rts-ops` (the exact path VSCode Remote-SSH uses)
reaches the box over IAP and returns a shell as `rtsmapping_woodwellclimate_org`.

### Retired: `vmup.ps1`

The old workflow started a VM, read back its new ephemeral external IP, and rewrote the SSH
config — plus a periodic re-authorization of your source IP. IAP removes the reason for all
of it. `vmup.ps1` is no longer used; delete your desktop shortcut.

## 4. What runs where

| Work | Runs on | Why |
|---|---|---|
| Editing code, git, docs | `ARCHITECTURE` | It is the control node |
| `gcloud` lifecycle, quota checks, billing | `ARCHITECTURE` | No reason to be inside the cloud |
| Planet acquisition (`run_year.sh`) | `rts-ops`, in `tmux` | Runs for days; shared with Heidi |
| S2 export driver | `rts-ops`, in `tmux` | Polls GEE for days |
| Cron alerters | `rts-ops` | Must fire every 10 min regardless of your desk |
| Rating app | `rts-review` | Public `:80`, must stay up |

If you find yourself about to `nohup` something on `ARCHITECTURE`, that is the signal it
belongs on `rts-ops`.

## 5. Cost discipline

| Host | Running | Notes |
|---|---|---|
| `rts-ops` (`e2-standard-2`) | ~$49/mo | Stop it whenever both loops are idle |
| `rts-review` (`e2-small`) | ~$12/mo | Leave up while the campaign is open |

There is no expiring credit any more — `abruptthawmapping` bills a real Woodwell account.
An idle GPU VM is no longer a wasted allowance; it is a bill.

## 6. Sharing `rts-ops`

Heidi runs Planet acquisition on the same box. Conventions that keep that painless:

- **The repo checkout is shared and read-only.** Nothing writes into it. Outputs go to
  `/mnt/outputs/`.
- **Credentials are per-user.** Each person's ADC lives in its own directory
  (`CLOUDSDK_CONFIG=/mnt/outputs/adc-<name> gcloud auth application-default login`), and is
  readable by anyone with sudo on the box — the same caveat as the Planet API key. If that
  is not acceptable for a given credential, run that job from your own machine instead.
- **Name your `tmux` session** after the job (`tmux new -s planet`), so an attached session
  is self-explanatory.
- **Never branch-switch the ops checkout.** It is mounted live into the running containers and
  read by cron; changing branches under a multi-day job swaps its source out from under it.
  Develop on `ARCHITECTURE`, and update `rts-ops` with a deliberate `git pull` on one branch.
  This is not hypothetical — see `pdg_migration.md` §4a.
