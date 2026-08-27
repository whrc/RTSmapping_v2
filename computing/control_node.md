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

1. **Google Cloud SDK** — <https://cloud.google.com/sdk/docs/install>. Then, in PowerShell:

   ```powershell
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project abruptthawmapping
   gcloud config set compute/zone us-west1-a
   ```

   Two logins, not one. `auth login` is you at the CLI; `application-default login` writes
   the credential that Python libraries (`google-cloud-storage`, `earthengine-api`) pick up.

2. **VSCode** with the **Remote - SSH** extension.

3. **Confirm IAP works** before relying on it:

   ```powershell
   gcloud compute ssh rts-ops --tunnel-through-iap
   ```

   A `4033 not authorized` means the `roles/iap.tunnelResourceAccessor` grant is missing —
   ask the project owner (Heidi). IAP TCP forwarding also needs the VPC to allow ingress
   from `35.235.240.0/20` on `tcp:22`; the default network's `default-allow-ssh` covers it.

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

Add this to `C:\Users\<you>\.ssh\config` (adjust the gcloud path if your install differs):

```
Host rts-ops
    HostName        rts-ops
    User            <your-os-login-username>
    IdentityFile    C:\Users\<you>\.ssh\google_compute_engine
    ProxyCommand    "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" compute start-iap-tunnel %h %p --listen-on-stdin --project=abruptthawmapping --zone=us-west1-a --verbosity=warning
    StrictHostKeyChecking no
```

Then **Ctrl+Shift+P → Remote-SSH: Connect to Host → `rts-ops`**. Your OS Login username is
printed by `gcloud compute os-login describe-profile --format="value(posixAccounts[0].username)"`.

The `google_compute_engine` key pair is created for you the first time you run
`gcloud compute ssh`. Run that once before configuring VSCode.

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
