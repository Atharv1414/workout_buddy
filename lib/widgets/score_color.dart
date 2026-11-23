import 'package:flutter/material.dart';

/// Colour for a 0-100 form score: green from 80, amber from 50, red below.
Color scoreColor(int score) => switch (score) {
  >= 80 => const Color(0xFF71E6A5),
  >= 50 => Colors.amber,
  _ => const Color(0xFFFF7A6B),
};
