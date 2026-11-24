# Gym subscriptions

Workout Buddy is sold to **gyms**, not to individual members. A gym pays you
monthly or annually; its members use the app free as part of their
membership. There is no server: each gym gets a **signed access code** that
the app checks on the phone.

## How it works

1. You and the gym agree a plan (monthly or annual) and you invoice them.
2. When they pay, you **issue a code** for them that runs until the end of
   the paid period.
3. The gym gives members the code in one of two ways:
   - **A button in the gym's own app or website** that opens
     `workoutbuddy://activate?code=…`. Members tap it once and Workout Buddy
     activates. This is the main integration.
   - **The code itself** (on a poster, QR code, email or at reception).
     Members paste it on the activation screen.
4. The app shows the gym's **name and brand colour** while the code is
   active.
5. When the paid period ends, members get **7 days' grace** with a "Your gym
   needs to renew" banner, then workouts lock until a new code arrives.
6. On renewal, issue a new code with a later date and send it to the gym. They
   update their button or poster. Members who open the new link or paste the
   new code are extended; an older code never shortens a newer one.

## One-time setup (done)

A signing key pair was created on 24 Nov 2026:

- **Private key:** `~/.workout_buddy/signing_key.json` on your Mac. **Back
  this file up now** (password manager or encrypted USB). Anyone with it can
  create codes. If you lose it, you can't issue codes that the released app
  accepts; you'd need a new key and an app update.
- **Public key:** `lib/services/license_keys.dart`. This one is safe to
  publish.
- A **demo code** (gym "Workout Buddy Demo", valid to 31 Dec 2027) is in
  `~/.workout_buddy/demo_code.txt`. Use it for your own testing, sales demos
  and App Store review. It's deliberately not in the repo.

Never commit the private key. If you sell the app, the private key goes to
the buyer as part of the handover.

## Issue a code for a gym

From the project folder:

```bash
dart run tool/gym_code.dart issue --id ironworks-pune --name "Ironworks Gym" --color FF7A00 --plan annual --until 2027-11-30
```

| Option | Meaning |
|---|---|
| `--id` | Short permanent ID for the gym: lowercase letters, digits, dashes. Keep it the same on renewals. |
| `--name` | Name shown in the app. |
| `--color` | Gym brand colour as hex (`RRGGBB`). The app builds its theme from it. |
| `--plan` | `monthly` or `annual` (shown in Settings). |
| `--until` | Last paid day, `YYYY-MM-DD`. Access lasts to the end of that day plus 7 days' grace. |

It prints the **code** and the **activation link**. Send both to the gym.
Keep a simple record of every code you issue: gym, plan, until, invoice
number.

**Monthly plans:** to save sending a new code every month, you can issue a
code a few days past the paid month and renew it each time they pay.

## Integration guide for the gym's app

Send this section to the gym's developer.

> Add a button such as "Form coach" that opens:
>
> `workoutbuddy://activate?code=<the code we sent you>`
>
> Always use the newest code we send you. If Workout Buddy isn't installed,
> send the member to the App Store or Google Play listing.
>
> - **iOS:** to check whether it's installed with `canOpenURL`, add
>   `workoutbuddy` to `LSApplicationQueriesSchemes` in your Info.plist.
>   Otherwise just call `open` and fall back to the store link if it fails.
> - **Android:** start an `ACTION_VIEW` intent with the link. Catch
>   `ActivityNotFoundException` and open the Play Store listing instead.
> - **Website / email:** a normal link works on phones with the app
>   installed.

## Suggested pricing

These are starting points, not market data. Adjust after your first few
gyms.

| Plan | Price | Notes |
|---|---|---|
| Pilot | Free for 1 month | In return for a testimonial and how many members used it |
| Monthly | $50–200 per gym location | Scale with member count |
| Annual | 10 × the monthly price | Two months free; easier for you to manage |
| Setup (optional) | $250–1,000 | Branding and help adding the button to their app |

## Limits of offline codes (know before you sell)

- **A code can be shared.** Anyone who gets a gym's code can use the app
  until that code ends. That's acceptable for pilots and small gyms. If a gym
  needs per-member control, the next step is a small server with member
  sign-in.
- **No usage numbers.** You can't see how many members activated. Ask the
  gym, or add analytics later.
- **The phone's clock is trusted.** A member who sets their clock back can
  extend access. Rare, and low risk at this price.
- **Codes can't be revoked early.** For a gym that stops paying mid-period,
  keep monthly gyms on short codes.

## App Store and Google Play review

- The app is locked without a code, so **reviewers need the demo code**. Put
  it in App Review Information → Notes (Apple) and in App access (Google
  Play), with: "Workout Buddy is provided to members by their gym. Paste this
  gym code on the first screen."
- The app sells nothing and shows no prices or purchase links. Gyms pay you
  directly, outside the app. Apple and Google can still question
  apps that unlock features without in-app purchase. Describe it clearly as a
  service sold to gyms (businesses) for their members. If a reviewer pushes
  back, the fallback is adding an optional in-app subscription for people who
  aren't gym members.
