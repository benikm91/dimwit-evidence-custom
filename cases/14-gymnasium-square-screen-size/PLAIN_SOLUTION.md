# Plain (NumPy) — verdict

**Category: `missed`**

Nothing here is a shape *error*. `preprocess_buggy(frame, 84)` returns an `84 x 84` array,
which is exactly what the program asked for. The defect is that the program should not have
asked for it: Atari frames are `210 x 160` and the aspect ratio is destroyed
(`test_the_aspect_ratio_is_destroyed`).

This is an **API expressiveness** fault. The parameter `screen_size: int` cannot say
"84 tall by 64 wide", so every user of the wrapper gets a squashed observation and has no
way to opt out. It is the kind of defect a type system reaches only if the type of the
parameter is wrong — and `int` is a perfectly good type for a number.
