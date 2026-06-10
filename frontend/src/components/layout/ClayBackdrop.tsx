export function ClayBackdrop() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden>
      <div className="animate-clay-float absolute -left-[10%] -top-[10%] h-[60vh] w-[60vh] rounded-full bg-[#8B5CF6]/10 blur-3xl" />
      <div className="animate-clay-float-delayed animation-delay-2000 absolute -right-[10%] top-[20%] h-[55vh] w-[55vh] rounded-full bg-[#EC4899]/10 blur-3xl" />
      <div className="animate-clay-float-slow animation-delay-4000 absolute bottom-[-10%] left-[20%] h-[50vh] w-[50vh] rounded-full bg-[#0EA5E9]/10 blur-3xl" />
    </div>
  );
}
