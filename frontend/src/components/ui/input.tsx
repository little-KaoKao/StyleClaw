import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-14 w-full min-w-0 rounded-2xl border-0 bg-clay-surface px-5 text-base text-foreground shadow-clayPressed transition-all duration-200 outline-none selection:bg-clay-accent selection:text-white file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
        "focus:bg-white focus:ring-4 focus:ring-clay-accent/20",
        "aria-invalid:ring-4 aria-invalid:ring-clay-accent-alt/30",
        className
      )}
      {...props}
    />
  )
}

export { Input }
