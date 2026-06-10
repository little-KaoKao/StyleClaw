import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-10 w-full min-w-0 rounded-none border-2 border-foreground bg-white px-3 py-1 text-base font-medium transition-[color,box-shadow] outline-none selection:bg-foreground selection:text-background file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
        "focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-foreground",
        "aria-invalid:border-bauhaus-red aria-invalid:ring-bauhaus-red",
        className
      )}
      {...props}
    />
  )
}

export { Input }
