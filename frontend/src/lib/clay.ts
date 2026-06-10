// Claymorphism shared class fragments.
export const CLAY_CARD = "rounded-[32px] bg-white/70 backdrop-blur-xl shadow-clayCard";
export const CLAY_CARD_HOVER = "transition-all duration-500 hover:-translate-y-2 hover:shadow-clayCardHover";
export const CLAY_BUTTON = "rounded-2xl shadow-clayButton transition-all duration-200 hover:-translate-y-1 hover:shadow-clayButtonHover active:scale-[0.92] active:shadow-clayPressed";
export const CLAY_PRESSED = "rounded-2xl shadow-clayPressed";
export const PRESS = "active:scale-[0.92] active:shadow-clayPressed transition-all duration-200";
export const LIFT = "transition-all duration-500 hover:-translate-y-2";
export const PRIMARY_GRADIENT = "bg-gradient-to-br from-[#A78BFA] to-[#7C3AED]";
// Candy accent colors for icon orbs / decorations (rotate through these)
export const ACCENTS = ["#7C3AED", "#DB2777", "#0EA5E9", "#10B981", "#F59E0B"] as const;
// Gradient pairs for icon orbs
export const ORB_GRADIENTS = [
  "from-purple-400 to-purple-600",
  "from-pink-400 to-pink-600",
  "from-sky-400 to-sky-600",
  "from-emerald-400 to-emerald-600",
  "from-amber-400 to-amber-600",
] as const;
